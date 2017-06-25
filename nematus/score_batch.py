"""
Given a parallel corpus of sentence pairs: with one-to-one of target and source sentences,
produce the score, and optionally alignment for each pair.
"""

import sys
import argparse
import tempfile

import os
import fnmatch

import numpy
import json

from data_iterator import TextIterator
from util import load_dict, load_config
from alignment_util import *
from compat import fill_options

from theano_util import (load_params, init_theano_params)
from nmt import (pred_probs, build_model, prepare_data, init_params)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano

def rescore_model(source_file, glob_pattern, target_file, saveto, models, options, b, normalization_alpha, verbose, alignweights):

    trng = RandomStreams(1234)

    fs_log_probs = []

    for model, option in zip(models, options):

        # load model parameters and set theano shared variables
        param_list = numpy.load(model).files
        param_list = dict.fromkeys([key for key in param_list if not key.startswith('adam_')], 0)
        params = load_params(model, param_list)
        tparams = init_theano_params(params)

        trng, use_noise, \
            x, x_mask, y, y_mask, \
            opt_ret, \
            cost = \
            build_model(tparams, option)
        inps = [x, x_mask, y, y_mask]
        use_noise.set_value(0.)

        if alignweights:
            sys.stderr.write("\t*** Save weight mode ON, alignment matrix will be saved.\n")
            outputs = [cost, opt_ret['dec_alphas']]
            f_log_probs = theano.function(inps, outputs)
        else:
            f_log_probs = theano.function(inps, cost)

        fs_log_probs.append(f_log_probs)

    def _score(pairs, alignweights=False):
        # sample given an input sequence and obtain scores
        scores = []
        alignments = []
        for i, f_log_probs in enumerate(fs_log_probs):
            score, alignment = pred_probs(f_log_probs, prepare_data, options[i], pairs, normalization_alpha=normalization_alpha, alignweights = alignweights)
            scores.append(score)
            alignments.append(alignment)

        return scores, alignments

    sfiles = get_files(source_file, glob_pattern)
    for sfilepath in sfiles:
        fbasename = os.path.basename(sfilepath)
        print fbasename
        tfilepath = os.path.join(target_file, fbasename)
        ofilepath = os.path.join(saveto, fbasename)
        with open(sfilepath, 'r') as sfile, open(tfilepath, 'r') as tfile, open(ofilepath, 'w') as ofile:
            pairs = TextIterator(
                sfile.name,
                tfile.name,
                options[0]['dictionaries'][:-1],
                options[0]['dictionaries'][-1],
                n_words_source=options[0]['n_words_src'],
                n_words_target=options[0]['n_words'],
                batch_size=b,
                maxlen=float('inf'),
                sort_by_length=False) #TODO: sorting by length could be more  efficient, but we'd want to resort after

            scores, alignments = _score(pairs, alignweights)

            sfile.seek(0)
            tfile.seek(0)
            source_lines = sfile.readlines()
            target_lines = tfile.readlines()

            for i, line in enumerate(target_lines):
                score_str = ' '.join(map(str,[s[i] for s in scores]))
                if verbose:
                    ofile.write('{0} '.format(line.strip()))
                ofile.write('{0}\n'.format(score_str))

            ### optional save weights mode.
            if alignweights:
                ### writing out the alignments.
#                 temp_name = os.path.splitext(fbasename)[0] + ".json"
                temp_name = ofile.name + ".json"
                with tempfile.NamedTemporaryFile(prefix=temp_name) as align_OUT:
#                     for line in all_alignments:
                    for line in alignments:
#                         align_OUT.write(line + "\n")
                        align_OUT.write("\n".join(line) + "\n")
                    ### combining the actual source and target words.
                    combine_source_target_text_1to1(sfile, tfile, ofile.name, align_OUT)


def get_files(directory, fileclue):
        """Get all files in a directory matching a pattern.

        Keyword arguments:
        directory -- a string for the input folder path
        fileclue -- a string as glob pattern
        """
        matches = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, fileclue):
                matches.append(os.path.join(root, filename))
        return matches


def main(models, source_file, glob_pattern, nbest_file, saveto, b=80,
         normalization_alpha=0.0, verbose=False, alignweights=False):

    # load model model_options
    options = []
    for model in models:
        options.append(load_config(model))

        fill_options(options[-1])

    if not os.path.exists(saveto):
        os.makedirs(saveto)

    rescore_model(source_file, glob_pattern, nbest_file, saveto, models, options, b, normalization_alpha, verbose, alignweights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=80,
                        help="Minibatch size (default: %(default)s))")
    parser.add_argument('-n', type=float, default=0.0, nargs="?", const=1.0, metavar="ALPHA",
                        help="Normalize scores by sentence length (with argument, exponentiate lengths by ALPHA)")
    parser.add_argument('-v', action="store_true", help="verbose mode.")
    parser.add_argument('--models', '-m', type=str, nargs = '+', required=True,
                        help="model to use. Provide multiple models (with same vocabulary) for ensemble decoding")
    parser.add_argument('--source', '-s',
                        required=True, metavar='PATH',
                        help="Source text file")
    parser.add_argument('--target', '-t',
                        required=True, metavar='PATH',
                        help="Target text file")
    parser.add_argument('--output', '-o',
                        default=sys.stdout, metavar='PATH',
                        help="Output file (default: standard output)")
    parser.add_argument('--walign', '-w',required = False,action="store_true",
                        help="Whether to store the alignment weights or not. If specified, weights will be saved in <target>.alignment")
    parser.add_argument('--glob_pattern', '-g', type=str, required=False, default="*.txt",
                        help="glob pattern to select input files.")

    args = parser.parse_args()

    main(args.models, args.source, args.glob_pattern, args.target,
         args.output, b=args.b, normalization_alpha=args.n, verbose=args.v, alignweights=args.walign)

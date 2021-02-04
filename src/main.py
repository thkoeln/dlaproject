import argparse
import os
import trainers as trainer_import
from trainers import music_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', type=str, default='Music',
                        help='Trainer: Music')
    parser.add_argument('--epochs', type=int, default=10) # Test with 120
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lstm_layers', type=int, default=64)
    parser.add_argument('--composer', type=str, default="albeniz")
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--plot', default=True, action='store_true', help='plot loss')

    config = parser.parse_args()

    # generate Trainer
    prep = getattr(trainer_import, config.trainer.lower() + "_trainer")
    prepare_fct = getattr(prep, 'Trainer' + config.trainer)
    trainer = prepare_fct(**config.__dict__)

    # run training
    trainer.train(**config.__dict__)

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()

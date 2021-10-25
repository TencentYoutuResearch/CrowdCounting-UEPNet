from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visualization parameters
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--eval_freq', type=int, default=2, help='frequency of epochs to perform evaluation on validation set')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=500, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', default=0.9, type=float, help='beta1 fro adam optimizer')
        parser.add_argument('--beta2', default=0.999, type=float, help='beta2 fro adam optimizer')
        parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=5e10, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser

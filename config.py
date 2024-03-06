class ConfigArgsPCT:
    def __init__(self, exp_name='exp', batch_size=512, epochs=100, lr=0.1, seed=1,
                 use_wandb=0, use_second_deg=0, lpe_normalize=0, use_pct=0,
                 std_dev=0, use_mlp=0, lpe_dim=3, use_xyz=1, num_of_heads=1,
                 num_neurons_per_layer=64, num_mlp_layers=4, num_of_attention_layers=1,
                 att_per_layer=4, output_dim=4, lr_jumps=50, sampled_points=40,
                 PE_dim=0):
        self.exp_name = exp_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.use_wandb = use_wandb
        self.use_second_deg = use_second_deg
        self.lpe_normalize = lpe_normalize
        self.use_pct = use_pct
        self.std_dev = std_dev
        self.use_mlp = use_mlp
        self.lpe_dim = lpe_dim
        self.use_xyz = use_xyz
        self.num_of_heads = num_of_heads
        self.num_neurons_per_layer = num_neurons_per_layer
        self.num_mlp_layers = num_mlp_layers
        self.num_of_attention_layers = num_of_attention_layers
        self.att_per_layer = att_per_layer
        self.output_dim = output_dim
        self.lr_jumps = lr_jumps
        self.sampled_points = sampled_points
        self.PE_dim = PE_dim
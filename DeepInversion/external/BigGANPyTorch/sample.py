''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses

from matplotlib import pyplot as plt

def run(config):
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
  # print(config)
  # exit()
  # Optionally, get the configuration from the state dict. This allows for
  # recovery of the config provided only a state dict and experiment name,
  # and can be convenient for writing less verbose sample shell scripts.
  if config['config_from_name']:
    # print(config['weights_root'],config['experiment_name'], config['load_weights'])
    utils.load_weights(None, None, state_dict, config['weights_root'], 
                       config['experiment_name'], config['load_weights'], None,
                       strict=False, load_optim=False)
    # Ignore items which we might want to overwrite from the command line
    for item in state_dict['config']:
      if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
        config[item] = state_dict['config'][item]
  
  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  config = utils.update_config_roots(config)
  config['skip_init'] = True
  config['no_optim'] = True
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])
   
  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True
  
  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)
  
  G = model.Generator(**config).cuda()

  # zht: my code
  # D = model.Discriminator(**config).cuda()
  from torch.nn import ReLU
  config_fixed = {'dataset': 'I128_hdf5', 'augment': False, 'num_workers': 0, 'pin_memory': True, 'shuffle': True, 'load_in_mem': False, 'use_multiepoch_sampler': True, 'model': 'BigGAN', 'G_param': 'SN', 'D_param': 'SN', 'G_ch': 96, 'D_ch': 96, 'G_depth': 1, 'D_depth': 1, 'D_wide': True, 'G_shared': True, 'shared_dim': 128, 'dim_z': 120, 'z_var': 1.0, 'hier': True, 'cross_replica': False, 'mybn': False, 'G_nl': 'inplace_relu', 'D_nl': 'inplace_relu', 'G_attn': '64', 'D_attn': '64', 'norm_style': 'bn', 'seed': 0, 'G_init': 'ortho', 'D_init': 'ortho', 'skip_init': True, 'G_lr': 0.0001, 'D_lr': 0.0004, 'G_B1': 0.0, 'D_B1': 0.0, 'G_B2': 0.999, 'D_B2': 0.999, 'batch_size': 256, 'G_batch_size': 64, 'num_G_accumulations': 8, 'num_D_steps': 1, 'num_D_accumulations': 8, 'split_D': False, 'num_epochs': 100, 'parallel': True, 'G_fp16': False, 'D_fp16': False, 'D_mixed_precision': False, 'G_mixed_precision': False, 'accumulate_stats': False, 'num_standing_accumulations': 16, 'G_eval_mode': True, 'save_every': 1000, 'num_save_copies': 2, 'num_best_copies': 5, 'which_best': 'IS', 'no_fid': False, 'test_every': 2000, 'num_inception_images': 50000, 'hashname': False, 'base_root': '', 'data_root': 'data', 'weights_root': 'weights', 'logs_root': 'logs', 'samples_root': 'samples', 'pbar': 'mine', 'name_suffix': '', 'experiment_name': '', 'config_from_name': False, 'ema': True, 'ema_decay': 0.9999, 'use_ema': True, 'ema_start': 20000, 'adam_eps': 1e-06, 'BN_eps': 1e-05, 'SN_eps': 1e-06, 'num_G_SVs': 1, 'num_D_SVs': 1, 'num_G_SV_itrs': 1, 'num_D_SV_itrs': 1, 'G_ortho': 0.0, 'D_ortho': 0.0, 'toggle_grads': True, 'which_train_fn': 'GAN', 'load_weights': '', 'resume': False, 'logstyle': '%3.3e', 'log_G_spectra': False, 'log_D_spectra': False, 'sv_log_interval': 10, 'sample_npz': True, 'sample_num_npz': 50000, 'sample_sheets': True, 'sample_interps': True, 'sample_sheet_folder_num': -1, 'sample_random': True, 'sample_trunc_curves': '0.05_0.05_1.0', 'sample_inception_metrics': True, 'resolution': 128, 'n_classes': 1000, 'G_activation': ReLU(inplace=True), 'D_activation': ReLU(inplace=True), 'no_optim': True}
  # config_fixed = {'dataset': 'I128_hdf5', 'augment': False, 'num_workers': 0, 'pin_memory': True, 'shuffle': True, 'load_in_mem': False, 'use_multiepoch_sampler': True, 'model': 'BigGAN', 'G_param': 'SN', 'D_param': 'SN', 'G_ch': 96, 'D_ch': 96, 'G_depth': 1, 'D_depth': 1, 'D_wide': True, 'G_shared': True, 'shared_dim': 128, 'dim_z': 120, 'z_var': 1.0, 'hier': True, 'cross_replica': False, 'mybn': False, 'G_nl': 'inplace_relu', 'D_nl': 'inplace_relu', 'G_attn': '64', 'D_attn': '64', 'norm_style': 'bn', 'seed': 0, 'G_init': 'ortho', 'D_init': 'ortho', 'skip_init': True, 'G_lr': 0.0001, 'D_lr': 0.0004, 'G_B1': 0.0, 'D_B1': 0.0, 'G_B2': 0.999, 'D_B2': 0.999, 'batch_size': 256, 'G_batch_size': 64, 'num_G_accumulations': 8, 'num_D_steps': 1, 'num_D_accumulations': 8, 'split_D': False, 'num_epochs': 100, 'parallel': True, 'G_fp16': False, 'D_fp16': False, 'D_mixed_precision': False, 'G_mixed_precision': False, 'accumulate_stats': False, 'num_standing_accumulations': 16, 'G_eval_mode': True, 'save_every': 1000, 'num_save_copies': 2, 'num_best_copies': 5, 'which_best': 'IS', 'no_fid': False, 'test_every': 2000, 'num_inception_images': 50000, 'hashname': False, 'base_root': '', 'data_root': 'data', 'weights_root': 'weights', 'logs_root': 'logs', 'samples_root': 'samples', 'pbar': 'mine', 'name_suffix': '', 'experiment_name': '', 'config_from_name': False, 'ema': True, 'ema_decay': 0.9999, 'use_ema': True, 'ema_start': 20000, 'adam_eps': 1e-06, 'BN_eps': 1e-05, 'SN_eps': 1e-06, 'num_G_SVs': 1, 'num_D_SVs': 1, 'num_G_SV_itrs': 1, 'num_D_SV_itrs': 1, 'G_ortho': 0.0, 'D_ortho': 0.0, 'toggle_grads': True, 'which_train_fn': 'GAN', 'load_weights': '', 'resume': False, 'logstyle': '%3.3e', 'log_G_spectra': False, 'log_D_spectra': False, 'sv_log_interval': 10, 'sample_npz': True, 'sample_num_npz': 50000, 'sample_sheets': True, 'sample_interps': True, 'sample_sheet_folder_num': -1, 'sample_random': True, 'sample_trunc_curves': '0.05_0.05_1.0', 'sample_inception_metrics': True, 'resolution': 128, 'n_classes': 1000, 'no_optim': True}
  D = model.Discriminator(**config_fixed).cuda()
  utils.load_weights(None, D, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     None,
                     strict=False, load_optim=False)
  D.eval()

  utils.count_parameters(G)
  
  # Load weights
  print('Loading weights...')
  # Here is where we deal with the ema--load ema weights or load normal weights
  utils.load_weights(G if not (config['use_ema']) else None, None, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     G if config['ema'] and config['use_ema'] else None,
                     strict=False, load_optim=False)
  # Update batch size setting used for G
  G_batch_size = max(config['G_batch_size'], config['batch_size']) 
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'], 
                             z_var=config['z_var'])
  
  if config['G_eval_mode']:
    print('Putting G in eval mode..')
    G.eval()
  else:
    print('G is in %s mode...' % ('training' if G.training else 'eval'))
    
  #Sample function
  sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)  
  if config['accumulate_stats']:
    print('Accumulating standing stats across %d accumulations...' % config['num_standing_accumulations'])
    utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                    config['num_standing_accumulations'])
    
  
  # Sample a number of images and save them to an NPZ, for use with TF-Inception
  if config['sample_npz']:
    # Lists to hold images and labels for images
    x, y = [], []
    print('Sampling %d images and saving them to npz...' % config['sample_num_npz'])
    for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
      with torch.no_grad():
        images, labels = sample()
      # zht: show discriminator results
      print(images.size(), labels.size())
      dis_loss = D(x=images, y=labels)
      print(dis_loss.size())
      print(dis_loss)
      exit()


      x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
      y += [labels.cpu().numpy()]

      plt.imshow(x[0][i,:,:,:].transpose((1,2,0)))
      plt.show()

    x = np.concatenate(x, 0)[:config['sample_num_npz']]
    y = np.concatenate(y, 0)[:config['sample_num_npz']]    
    print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
    npz_filename = '%s/%s/samples.npz' % (config['samples_root'], experiment_name)
    print('Saving npz to %s...' % npz_filename)
    np.savez(npz_filename, **{'x' : x, 'y' : y})
  
  # Prepare sample sheets
  if config['sample_sheets']:
    print('Preparing conditional sample sheets...')
    utils.sample_sheet(G, classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']], 
                         num_classes=config['n_classes'], 
                         samples_per_class=10, parallel=config['parallel'],
                         samples_root=config['samples_root'], 
                         experiment_name=experiment_name,
                         folder_number=config['sample_sheet_folder_num'],
                         z_=z_,)
  # Sample interp sheets
  if config['sample_interps']:
    print('Preparing interp sheets...')
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
      utils.interp_sheet(G, num_per_sheet=16, num_midpoints=8,
                         num_classes=config['n_classes'], 
                         parallel=config['parallel'], 
                         samples_root=config['samples_root'], 
                         experiment_name=experiment_name,
                         folder_number=config['sample_sheet_folder_num'], 
                         sheet_number=0,
                         fix_z=fix_z, fix_y=fix_y, device='cuda')
  # Sample random sheet
  if config['sample_random']:
    print('Preparing random sample sheet...')
    images, labels = sample()    
    torchvision.utils.save_image(images.float(),
                                 '%s/%s/random_samples.jpg' % (config['samples_root'], experiment_name),
                                 nrow=int(G_batch_size**0.5),
                                 normalize=True)

  # Get Inception Score and FID
  get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'])
  # Prepare a simple function get metrics that we use for trunc curves
  def get_metrics():
    sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)    
    IS_mean, IS_std, FID = get_inception_metrics(sample, config['num_inception_images'], num_splits=10, prints=False)
    # Prepare output string
    outstring = 'Using %s weights ' % ('ema' if config['use_ema'] else 'non-ema')
    outstring += 'in %s mode, ' % ('eval' if config['G_eval_mode'] else 'training')
    outstring += 'with noise variance %3.3f, ' % z_.var
    outstring += 'over %d images, ' % config['num_inception_images']
    if config['accumulate_stats'] or not config['G_eval_mode']:
      outstring += 'with batch size %d, ' % G_batch_size
    if config['accumulate_stats']:
      outstring += 'using %d standing stat accumulations, ' % config['num_standing_accumulations']
    outstring += 'Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID)
    print(outstring)
  if config['sample_inception_metrics']: 
    print('Calculating Inception metrics...')
    get_metrics()
    
  # Sample truncation curve stuff. This is basically the same as the inception metrics code
  if config['sample_trunc_curves']:
    start, step, end = [float(item) for item in config['sample_trunc_curves'].split('_')]
    print('Getting truncation values for variance in range (%3.3f:%3.3f:%3.3f)...' % (start, step, end))
    for var in np.arange(start, end + step, step):     
      z_.var = var
      # Optionally comment this out if you want to run with standing stats
      # accumulated at one z variance setting
      if config['accumulate_stats']:
        utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                    config['num_standing_accumulations'])
      get_metrics()
def main():
  # parse command line and run    
  parser = utils.prepare_parser()
  parser = utils.add_sample_parser(parser)
  config = vars(parser.parse_args())
  print(config)
  run(config)
  
if __name__ == '__main__':    
  main()
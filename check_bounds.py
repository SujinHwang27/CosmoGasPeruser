import numpy as np
params = np.load('data/feature_discovery/base_data/micro_classifier_params.npy')
print('Decisions: Min=', np.min(params[:, :24]), 'Max=', np.max(params[:, :24]))
print('Intercepts: Min=', np.min(params[:, 24:]), 'Max=', np.max(params[:, 24:]))

# v pred
python train_upsampling.py --config configs/mink_base.yml --diffusion.objective pred_v --diffusion.schedule cosine --name mink_pred_v_cosine
python train_upsampling.py --config configs/mink_base.yml --diffusion.objective pred_v --diffusion.schedule linear --name mink_pred_v_linear
python train_upsampling.py --config configs/mink_base.yml --diffusion.objective pred_v --diffusion.schedule sigmoid --name mink_pred_v_sigmoid
# noise pred
python train_upsampling.py --config configs/mink_base.yml --diffusion.objective pred_noise --diffusion.schedule cosine --name mink_pred_noise_cosine
python train_upsampling.py --config configs/mink_base.yml --diffusion.objective pred_noise --diffusion.schedule linear --name mink_pred_noise_linear
python train_upsampling.py --config configs/mink_base.yml --diffusion.objective pred_noise --diffusion.schedule sigmoid --name mink_pred_noise_sigmoid
# pred_x0
python train_upsampling.py --config configs/mink_base.yml --diffusion.objective pred_x0 --diffusion.schedule cosine --name mink_pred_x0_cosine
python train_upsampling.py --config configs/mink_base.yml --diffusion.objective pred_x0 --diffusion.schedule linear --name mink_pred_x0_linear
python train_upsampling.py --config configs/mink_base.yml --diffusion.objective pred_x0 --diffusion.schedule sigmoid --name mink_pred_x0_sigmoid
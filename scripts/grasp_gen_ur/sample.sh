CKPT=$1
OPT=$2

if [ -z ${CKPT} ]
then
    echo "No ckpt input."
    exit
fi

if [ -z ${OPT} ] || [ ${OPT} != "OPT" ]
then
    echo -e "\033[1;38;2;255;165;0m[WITHOUT] Running without Physics-Guided Sampling\033[0m"
    python sample.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT} \
                diffuser=ddpm \
                diffuser.loss_type=l1 \
                diffuser.steps=100 \
                model=unet_grasp \
                task=grasp_gen_ur \
                task.dataset.normalize_x=true \
                task.dataset.normalize_x_trans=true
else
    echo -e "\033[1;32m[WITH] Physics-Guided Sampling Activated\033[0m"
    python sample.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT} \
                diffuser=ddpm \
                diffuser.loss_type=l1 \
                diffuser.steps=100 \
                model=unet_grasp \
                task=grasp_gen_ur \
                task.dataset.normalize_x=true \
                task.dataset.normalize_x_trans=true \
                optimizer=grasp_with_object 
fi
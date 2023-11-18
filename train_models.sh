#!/bin/bash

# Download and unzip the file
wget https://cims.nyu.edu/~brenden/supplemental/BIML-large-files/data_algebraic.zip
unzip data_algebraic.zip

# Create a new Python environment
mkdir -p /env/MCL1
python3 -m venv /env/MCL1

# Activate the environment and update the PATH
source /env/MCL1/bin/activate
export PATH="/transformers/src:$PATH"

# Clone the Transformers repository
git clone https://github.com/huggingface/transformers.git /transformers/
cd /transformers
git checkout v4.35.0

# Install the Transformers package
pip install git+https://github.com/huggingface/transformers.git@v4.35.0
pip install torch
pip install scikit-learn
pip install matplotlib

cd /MLC1/
# Run Python scripts 
python train_decoder.py --nepochs 50 --nlayers_decoder 1 --batch_size 25 --episode_type algebraic --fn_out_model net-LLAMA_50ep_1layer_alg.pt >/log1.txt 2>&1 &
python train_decoder.py --nepochs 50 --nlayers_decoder 2 --batch_size 25 --episode_type algebraic --fn_out_model net-LLAMA_50ep_2layer_alg.pt >/log2.txt 2>&1 &
wait
python train_decoder.py --nepochs 50 --nlayers_decoder 3 --batch_size 25 --episode_type algebraic --fn_out_model net-LLAMA_50ep_3layer_alg.pt >/log3.txt 2>&1 &
python train_decoder.py --nepochs 50 --nlayers_decoder 4 --batch_size 25 --episode_type algebraic --fn_out_model net-LLAMA_50ep_4layer_alg.pt >/log4.txt 2>&1 &
wait
python train_decoder.py --nepochs 50 --nlayers_decoder 4 --batch_size 25 --episode_type algebraic --fn_out_model net-LLAMA_50ep_5layer_alg.pt >/log5.txt 2>&1 &
wait
python eval_llama.py --episode_type algebraic --sample_html --fn_out_model net-LLAMA_50ep_1layer_alg.pt >eval/net-LLAMA_50ep_1layer_alg.txt
python eval_llama.py --episode_type algebraic --sample_html --fn_out_model net-LLAMA_50ep_2layer_alg.pt >eval/net-LLAMA_50ep_2layer_alg.txt
python eval_llama.py --episode_type algebraic --sample_html --fn_out_model net-LLAMA_50ep_3layer_alg.pt >eval/net-LLAMA_50ep_3layer_alg.txt
python eval_llama.py --episode_type algebraic --sample_html --fn_out_model net-LLAMA_50ep_4layer_alg.pt >eval/net-LLAMA_50ep_4layer_alg.txt
python eval_llama.py --episode_type algebraic --sample_html --fn_out_model net-LLAMA_50ep_5layer_alg.pt >eval/net-LLAMA_50ep_5layer_alg.txt

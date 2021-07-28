from speaker_encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from speaker_encoder.params_model import *
from speaker_encoder.model import SpeakerEncoder
from pathlib import Path
import torch
import sys
import random
import numpy as np


def get_cossim(src_embeds, tgt_embeds, loss_device):
    speakers_per_batch, utterances_per_speaker = src_embeds.shape[:2]
    
    # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
    centroids_incl = torch.mean(tgt_embeds, dim=1, keepdim=True)
    centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

    # Exclusive centroids (1 per utterance)
    centroids_excl = (torch.sum(tgt_embeds, dim=1, keepdim=True) - tgt_embeds)
    centroids_excl /= (utterances_per_speaker - 1)
    centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

    # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
    # product of these vectors (which is just an element-wise multiplication reduced by a sum).
    # We vectorize the computation for efficiency.
    sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                speakers_per_batch).to(loss_device)
    mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
    for j in range(speakers_per_batch):
        mask = np.where(mask_matrix[j])[0]
        sim_matrix[mask, :, j] = (src_embeds[mask] * centroids_incl[j]).sum(dim=2)
        sim_matrix[j, :, j] = (src_embeds[j] * centroids_excl[j]).sum(dim=1)
    
    return sim_matrix

def test(clean_data_root: Path, model_path: Path, ):
    # Create a dataset and a dataloader
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=8,
    )

    # Setup the device on which to run the forward pass and the loss. These can be different, 
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")

    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        print("No model \"%s\" found, Exiting." % model_path)
        sys.exit()

    model.eval()

    avg_EER = 0
    for e in range(1):
        batch_avg_EER = 0
        for step, speaker_batch in enumerate(loader):
            assert utterances_per_speaker % 2 == 0
            
            inputs = speaker_batch.data
            
            enrollment_idx = []
            verification_idx = []
            idx = 0
            for spk_idx in range(speakers_per_batch):
                for utter_idx in range(utterances_per_speaker//2):
                    enrollment_idx.append(idx)
                    idx += 1
                for utter_idx in range(utterances_per_speaker//2):
                    verification_idx.append(idx)
                    idx += 1
                    
            enrollment_inputs = inputs[enrollment_idx]
            verification_inputs = inputs[verification_idx]
            
            enrollment_batch = torch.from_numpy(enrollment_inputs).to(device)
            verification_batch = torch.from_numpy(verification_inputs).to(device)

            perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
                
            verification_batch = verification_batch[perm]
            enrollment_embeddings = model(enrollment_batch)
            verification_embeddings = model(verification_batch)
            verification_embeddings = verification_embeddings[unperm]
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, (speakers_per_batch, utterances_per_speaker//2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (speakers_per_batch, utterances_per_speaker//2, verification_embeddings.size(1)))
            
            sim_matrix = get_cossim(verification_embeddings, enrollment_embeddings, loss_device)
            # sim_matrix = sim_matrix * model.similarity_weight + model.similarity_bias
            
            # calculating EER
            diff = 1; EER=1; EER_thresh = 0; EER_FAR=1; EER_FRR=0
            
            # print(sim_matrix)
            
            for thres in [0.01*i+0.5 for i in range(50)]:
                sim_matrix_thresh = sim_matrix>thres
                
                FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(speakers_per_batch))])
                /(speakers_per_batch-1.0)/(float(utterances_per_speaker/2))/speakers_per_batch)
    
                FRR = (sum([utterances_per_speaker/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(speakers_per_batch))])
                /(float(utterances_per_speaker/2))/speakers_per_batch)
                
                # Save threshold when FAR = FRR (=EER)
                if diff> abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
                
                # print("thres:%0.2f, FAR:%0.2f, FRR:%0.2f"%(thres,FAR,FRR))
                
            batch_avg_EER += EER
            print("Step: %d EER : %0.4f (thres:%0.2f, FAR:%0.4f, FRR:%0.4f)"%(step,EER,EER_thresh,EER_FAR,EER_FRR))
        avg_EER += batch_avg_EER/(batch_id+1)
    avg_EER = avg_EER / 1
    print("\n EER across {0} epochs: {1:.4f}".format(1, avg_EER))
    
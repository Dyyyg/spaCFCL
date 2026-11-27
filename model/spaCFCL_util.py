import torch
from tqdm import tqdm
import torch.nn.functional as F
from .modelTriatt_Flow1 import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing
from .utils import add_noise_by_zeroing, add_noise_by_zeroing_columns, add_gaussian_noise
import random
import matplotlib.pyplot as plt


class Train_spaCFCL:
    def __init__(self, data,
                 embedding_genef, embedding_scgpt,
                 datatype='SPOTS', device=torch.device('cpu'),
                 random_seed=2024, learning_rate=0.0001,
                 weight_decay=0.00, epochs=600,
                 dim_input=3000, dim_output=64,
                 weight_factors=[1, 5, 1, 1, 5, 5, 3, 3, 0.4],
                 cl_temperature=0.1, neg_sample_ratio=50,
                 temp_annealing=0.99, temp_annealing_step=100):
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.embedding_genef = torch.from_numpy(embedding_genef).to(self.device)
        self.embedding_scgpt = torch.from_numpy(embedding_scgpt).to(self.device)

        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors

        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']

        self.adj = adjacent_matrix_preprocessing(
            self.adata_omics1, self.adata_omics2,
            self.data['adj_emb_genef'],
            self.data['adj_emb_scgpt'])
      
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        # self.adj_spatial_omics2: protein spatial graph
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        # self.adj_feature_omics1: RNA feature graph
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        # self.adj_feature_omics2: protein feature graph
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        # self.adj_emb_genef: Geneformer embedding graph
        self.adj_emb_genef = self.adj['adj_emb_genef'].to(self.device)
        # self.adj_emb_scgpt: scGPT embedding graph
        self.adj_emb_scgpt = self.adj['adj_emb_scgpt'].to(self.device)
        # self.features_omics1: RNA features
        self.features_omics1 = torch.FloatTensor(
            self.adata_omics1.obsm['feat'].copy()).to(self.device)
        # self.features_omics2: protein features
        self.features_omics2 = torch.FloatTensor(
            self.adata_omics2.obsm['feat'].copy()).to(self.device)
        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs
        # self.dim_input1: RNA feature dimension
        self.dim_input1 = self.features_omics1.shape[1]
        # self.dim_input2: protein feature dimension
        self.dim_input2 = self.features_omics2.shape[1]
        # self.dim_output1: RNA output dimension
        self.dim_output1 = self.dim_output
        # self.dim_output2: protein output dimension
        self.dim_output2 = self.dim_output
        self.loss_history = []

        # contrastive learning loss
        from .CL import CL
        self.cl_loss_fn = CL(temperature=cl_temperature,
                             neg_sample_ratio=neg_sample_ratio,
                             temp_annealing=temp_annealing,
                             temp_annealing_step=temp_annealing_step).to(self.device)

        # if self.datatype == 'SPOTS':
        #     self.epochs = 600
        #     self.weight_factors = [1, 5, 1, 1, 5, 5, 3, 3, 0.4]
        # elif self.datatype == 'Stereo-CITE-seq':
        #     self.epochs = 1500
        #     self.weight_factors = [1, 10, 1, 10, 5, 5, 3, 3, 0.4]
        # elif self.datatype == '10x':
        #     self.epochs = 200
        #     # [5, 5, 1, 10, 10, 10, 10, 10, 6]     
        #     self.weight_factors = [5, 5, 1, 10, 10, 10, 10, 10, 6]   
        # elif self.datatype == 'Spatial-epigenome-transcriptome':
        #     self.epochs = 1600
        #     self.weight_factors = [1, 5, 1, 1, 10, 10, 5, 5, 0.4]

        # fallback: ensure len(weight_factors) == 9
        if len(self.weight_factors) < 9:
            self.weight_factors = self.weight_factors + [0] * (9 - len(self.weight_factors))

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.show()

    def train(self, epochs=100):
        # self.dim_input1: RNA feature dimension
        # self.dim_output1: RNA output dimension
        # self.dim_input2: protein feature dimension
        # self.dim_output2: protein output dimension
        # self.embedding_genef.shape[1]: Geneformer embedding dimension
        # self.embedding_scgpt.shape[1]: scGPT embedding dimension
        # self.dropout: dropout rate
        # self.act: activation function
        # Lock the RNG first so that model weight initialization is reproducible
        import torch as _torch
        _torch.manual_seed(self.random_seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(self.random_seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False

        self.model = Encoder_overall(
            self.dim_input1, self.dim_output1,
            self.dim_input2, self.dim_output2,
            dim_emb=self.embedding_genef.shape[1],
            dropout=0.00, act=F.relu).to(self.device)
        # self.optimizer: optimizer
        # self.learning_rate: learning rate
        # self.weight_decay: weight decay
        # self.model.parameters(): model parameters
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         self.learning_rate,
                                         weight_decay=self.weight_decay)
        self.model.train()

        print("epochs: ", epochs)
        for epoch in tqdm(range(epochs)):
            self.model.train()

            if random.random() < 0:
                features_omics1_noisy = add_gaussian_noise(
                    self.features_omics1, mean=0, std=0.1)
                cellembedding_noisey = add_gaussian_noise(
                    self.embedding_genef, mean=0, std=0.01)
                results = self.model(
                    features_omics1_noisy, self.features_omics2,
                    self.adj_spatial_omics1, self.adj_feature_omics1,
                    self.adj_spatial_omics2, self.adj_feature_omics2,
                    cellembedding_noisey, self.embedding_scgpt,
                    self.adj_emb_genef, self.adj_emb_scgpt)
            else:
                # self.features_omics1: RNA features
                # self.features_omics2: protein features
                # self.adj_spatial_omics1: RNA spatial graph
                # self.adj_feature_omics1: RNA feature graph
                # self.adj_spatial_omics2: protein spatial graph
                # self.adj_feature_omics2: protein feature graph
                # self.embedding_genef: Geneformer embedding
                # self.embedding_scgpt: scGPT embedding
                # self.adj_emb_genef: Geneformer embedding graph
                # self.adj_emb_scgpt: scGPT embedding graph
                results = self.model(
                    self.features_omics1, self.features_omics2,
                    self.adj_spatial_omics1, self.adj_feature_omics1,
                    self.adj_spatial_omics2, self.adj_feature_omics2,
                    self.embedding_genef, self.embedding_scgpt,
                    self.adj_emb_genef, self.adj_emb_scgpt)

            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(
                self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(
                self.features_omics2, results['emb_recon_omics2'])

            # LLM reconstruction loss (spa branch)
            self.loss_rec_genef = F.mse_loss(
                results['raw_emb_genef'], results['emb_recon_genef'])
            self.loss_rec_scgpt = F.mse_loss(
                results['raw_emb_scgpt'], results['emb_recon_scgpt'])

            # LLM reconstruction loss (fea branch; new: removes the blind spot
            # with no direct supervision)
            self.loss_rec_genef_fea = F.mse_loss(
                results['raw_emb_genef'], results['emb_recon_genef_fea'])
            self.loss_rec_scgpt_fea = F.mse_loss(
                results['raw_emb_scgpt'], results['emb_recon_scgpt_fea'])

            # correspondence loss
            self.loss_corr_omics1 = F.mse_loss(
                results['emb_latent_omics1'],
                results['emb_latent_omics1_across_recon'])
            self.loss_corr_omics2 = F.mse_loss(
                results['emb_latent_omics2'],
                results['emb_latent_omics2_across_recon'])

            # ========== InfoNCE contrastive learning loss ==========
            self.loss_cl = self.cl_loss_fn(
                results['emb_genef_spa_raw'],
                results['emb_scgpt_spa_raw'])

            # total loss (9 terms: r1, r2, c1, c2, g_spa, s_spa, g_fea, s_fea, cl)
            wf = self.weight_factors
            loss = (wf[0] * self.loss_recon_omics1 +
                    wf[1] * self.loss_recon_omics2 +
                    wf[2] * self.loss_corr_omics1 +
                    wf[3] * self.loss_corr_omics2 +
                    wf[4] * self.loss_rec_genef +
                    wf[5] * self.loss_rec_scgpt +
                    wf[6] * self.loss_rec_genef_fea +
                    wf[7] * self.loss_rec_scgpt_fea +
                    wf[8] * self.loss_cl)

            self.loss_history.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                cur_tau = self.cl_loss_fn.temperature.item()
                r1 = self.loss_recon_omics1.item()
                r2 = self.loss_recon_omics2.item()
                c1 = self.loss_corr_omics1.item()
                c2 = self.loss_corr_omics2.item()
                gs = self.loss_rec_genef.item()
                ss = self.loss_rec_scgpt.item()
                gf = self.loss_rec_genef_fea.item()
                sf = self.loss_rec_scgpt_fea.item()
                cl = self.loss_cl.item()
            print(
                f"epoch {epoch:4d} | total {loss.item():.4f} | "
                f"r1 {r1:.3f} r2 {r2:.3f} c1 {c1:.3f} c2 {c2:.3f} "
                f"g/spa {gs:.3f} s/spa {ss:.3f} g/fea {gf:.3f} s/fea {sf:.3f} "
                f"cl {cl:.3f} | tau {cur_tau:.4f}"
            )

        print("Model training finished!\n")

        with torch.no_grad():
            self.model.eval()
            results = self.model(
                self.features_omics1, self.features_omics2,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2,
                self.embedding_genef, self.embedding_scgpt,
                self.adj_emb_genef, self.adj_emb_scgpt)

        emb_omics1 = F.normalize(
            results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(
            results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(
            results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {
            'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
            'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
            'spaFDG': emb_combined.detach().cpu().numpy(),
            'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
            'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
            'alpha': results['alpha'].detach().cpu().numpy(),
            'alpha_att1': results['alpha_att1'].detach().cpu().numpy(),
            'alpha_att2': results['alpha_att2'].detach().cpu().numpy(),
            'alpha_llm_spa': results['alpha_llm_spa'].detach().cpu().numpy(),
            'alpha_llm_fea': results['alpha_llm_fea'].detach().cpu().numpy(),
            'final_temperature': self.cl_loss_fn.temperature.item(),
        }

        return output

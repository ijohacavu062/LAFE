import cv2
import numpy as np
import time
import torch
import pyiqa  
import os 

class LAFE_Optimizer:
    def __init__(self, use_gpu=True):
        self.device = torch.device("cpu")
        print(f"NIQE Evaluator running on: {self.device}")
        
        self.niqe_metric = pyiqa.create_metric('niqe').to(self.device)

    def compute_edge_intensity(self, I):
        Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(Ix**2 + Iy**2)
        grad_mag = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)
        return grad_mag

    def compute_phi_map_vectorized(self, WEk, a_global, b_global, block_size=8, gamma=0.5):
        h, w = WEk.shape
        # step1
        h_small, w_small = max(1, h // block_size), max(1, w // block_size)
        mean_edge_small = cv2.resize(WEk, (w_small, h_small), interpolation=cv2.INTER_AREA)
        
        # step2
        mean_edge_full = cv2.resize(mean_edge_small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # step3
        a_i = a_global * (1 + gamma * mean_edge_full)
        a_i = np.clip(a_i, 0, 3)
        
        phi_map = 1 + a_i * (WEk ** b_global)
        return phi_map

    def build_laplacian_pyramid(self, img, levels=6):
        pyramid = []
        current_img = img.copy()
        for i in range(levels - 1):
            down = cv2.pyrDown(current_img)
            # Strictly ensure that the size after upsample matches current_img
            up = cv2.pyrUp(down, dstsize=(current_img.shape[1], current_img.shape[0]))
            pyramid.append(current_img - up)
            current_img = down
        pyramid.append(current_img)
        return pyramid

    def build_gaussian_pyramid(self, img, levels=6):
        pyramid = [img.copy()]
        for i in range(levels - 1):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        return pyramid

    def multiscale_fusion(self, I1, I2, I3, W1, W2, W3, levels=6):
        L1 = self.build_laplacian_pyramid(I1, levels)
        L2 = self.build_laplacian_pyramid(I2, levels)
        L3 = self.build_laplacian_pyramid(I3, levels)
        
        G1 = self.build_gaussian_pyramid(W1, levels)
        G2 = self.build_gaussian_pyramid(W2, levels)
        G3 = self.build_gaussian_pyramid(W3, levels)
        
        # Layer-by-layer fusion
        fused_pyramid = []
        for l in range(levels):
            fused = G1[l]*L1[l] + G2[l]*L2[l] + G3[l]*L3[l]
            fused_pyramid.append(fused)
            
        # Reconstruction
        img = fused_pyramid[-1]
        for l in range(levels - 2, -1, -1):
            img = cv2.pyrUp(img, dstsize=(fused_pyramid[l].shape[1], fused_pyramid[l].shape[0]))
            img += fused_pyramid[l]
        return img

    def evaluate_niqe(self, image_rgb_01):
        #  PyTorch tensor, shape: (1, 3, H, W)
        img_tensor = torch.from_numpy(image_rgb_01).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            score = self.niqe_metric(img_tensor)
        return score.item()

    def enhance(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
        H = hsv_img[:, :, 0] / 180.0 * np.pi * 2  
        S = hsv_img[:, :, 1] / 255.0
        
        # === Step 1: Illumination Estimation ===
        L = np.max(img_rgb, axis=2)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) 
        I_estimated = cv2.morphologyEx(L, cv2.MORPH_CLOSE, se)
        
        # guided filtering
        guide = hsv_img[:, :, 2] / 255.0
        I_refined = self.manual_guided_filter(guide, I_estimated, window_size=31, eps=0.01)
        I_refined = np.clip(I_refined, 1e-4, 1.0)
        
        # R
        R = img_rgb / I_refined[:, :, np.newaxis]
        
        # === Step 2: Inputs Derivation ===
        I1 = I_refined.astype(np.float64)
        
        Imean = np.mean(I_refined)
        lambda_val = 10 + (1 - Imean) / Imean
        I2 = (2 / np.pi) * np.arctan(lambda_val * I_refined)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        I3 = clahe.apply((I_refined * 255).astype(np.uint8)) / 255.0
        
        WE1 = self.compute_edge_intensity(I1) ** 0.8
        WE2 = self.compute_edge_intensity(I2) ** 0.8
        WE3 = self.compute_edge_intensity(I3) ** 0.8
        
        # === Step 3: Base Weights ===
        def calc_WB(I): return np.exp(-((I - 0.5)**2) / (2 * 0.25**2))
        def calc_WC(I, S, H): return I * (1 + np.cos(2 * H - 250*np.pi/180) * S)
        
        WB1, WB2, WB3 = calc_WB(I1), calc_WB(I2), calc_WB(I3)
        WC1, WC2, WC3 = calc_WC(I1, S, H), calc_WC(I2, S, H), calc_WC(I3, S, H)
        
        # Basic weight combination
        W1_base = WB1 * WC1
        W2_base = WB2 * WC2
        W3_base = WB3 * WC3

        # === Step 4: Pattern Search ===
        def eval_fitness(a, b):
            phi1 = self.compute_phi_map_vectorized(WE1, a, b)
            phi2 = self.compute_phi_map_vectorized(WE2, a, b)
            phi3 = self.compute_phi_map_vectorized(WE3, a, b)
            
            W1 = W1_base * phi1
            W2 = W2_base * phi2
            W3 = W3_base * phi3
            
            W_sum = W1 + W2 + W3 + 1e-8
            W1_n, W2_n, W3_n = W1 / W_sum, W2 / W_sum, W3 / W_sum
            
            I_fused = self.multiscale_fusion(I1, I2, I3, W1_n, W2_n, W3_n)
            S_fused = np.clip(R * I_fused[:, :, np.newaxis], 0, 1)
            
            return self.evaluate_niqe(S_fused), S_fused

        print("Starting Pattern Search with Early Stopping...")
        # Initial parameters
        current_a, current_b = 1.0, 1.0
        step_size = 0.2
        best_score, best_img = eval_fitness(current_a, current_b)
        
        delta_history = []
        
        for iteration in range(1, 21): # T_max = 20
            candidates = [
                (current_a + step_size, current_b),
                (current_a - step_size, current_b),
                (current_a, current_b + step_size/2),
                (current_a, current_b - step_size/2)
            ]
            
            found_better = False
            for test_a, test_b in candidates:
                # lb=[0, 0.5], ub=[2, 2]
                test_a = np.clip(test_a, 0.0, 2.0)
                test_b = np.clip(test_b, 0.5, 2.0)
                
                score, img = eval_fitness(test_a, test_b)
                if score < best_score:
                    delta = best_score - score
                    delta_history.append(delta)
                    
                    best_score = score
                    best_img = img
                    current_a, current_b = test_a, test_b
                    found_better = True
                    break 
                    
            if not found_better:
                step_size *= 0.5 
                delta_history.append(0.0)
            
            print(f"Iter {iteration:02d}: NIQE = {best_score:.4f} | a = {current_a:.2f}, b = {current_b:.2f}")
            
            if len(delta_history) >= 3:
                d1, d2, d3 = delta_history[-3:]
                if abs(d3) < 1e-4 or (d1 * d2 < 0 and d2 * d3 < 0):
                    print(f"--> Early Stopping Triggered at iteration {iteration}!")
                    break

        final_bgr = cv2.cvtColor((best_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return final_bgr
    
    def manual_guided_filter(self, guide, src, window_size=31, eps=0.01):
        guide = guide.astype(np.float32)
        src = src.astype(np.float32)

        mean_I = cv2.blur(guide, (window_size, window_size))
        mean_p = cv2.blur(src, (window_size, window_size))
        mean_Ip = cv2.blur(guide * src, (window_size, window_size))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.blur(guide * guide, (window_size, window_size))
        var_I = mean_II - mean_I * mean_I

        # Calculate the coefficients a and b
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # Smooth out a and b
        mean_a = cv2.blur(a, (window_size, window_size))
        mean_b = cv2.blur(b, (window_size, window_size))

        q = mean_a * guide + mean_b
        return q

if __name__ == "__main__":
    lafe = LAFE_Optimizer()
    
    input_folder = 'raw'    
    output_folder = 'result' 

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG')
    image_files = [f for f in os.listdir(input_folder) if f.endswith(valid_extensions)]
    
    total_images = len(image_files)
    if total_images == 0:
        print(f"no pics!")
    else:
        
        start_all_time = time.time()
        
        for i, filename in enumerate(image_files):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            if img is not None:

                result = lafe.enhance(img)

                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, result)
            else:
                print(f"unable to process the picture: {filename}")
        
        end_all_time = time.time()
        
        total_duration = end_all_time - start_all_time
        avg_time = total_duration / total_images

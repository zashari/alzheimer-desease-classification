# =====================================================================
# ADNI MRI PROCESSING PIPELINE - STEP 8: GWO IMAGE ENHANCEMENT
# =====================================================================
# This module enhances the cropped 2D images using a Grey Wolf
# Optimizer (GWO) to find the optimal parameters for a chosen
# enhancement technique, applied only to the brain region.

from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import cv2
import numpy as np
from scipy import ndimage
from skimage.exposure import equalize_adapthist
from tqdm import tqdm

from configs import config
from utils import utils

# =====================================================================
# 1. MASK-AWARE ENHANCEMENT & QUALITY METRICS
# =====================================================================

class ImageEnhancer:
    """Collection of enhancement operators matching the notebook's implementation."""
    @staticmethod
    def adaptive_enhancement(img, clahe_clip=2.0, gabor_strength=0.3, unsharp_sigma=1.0, unsharp_strength=1.5):
        if img.max() > 1.0: img = img.astype(np.float32) / 255.0
        
        # CLAHE
        out = equalize_adapthist(img, clip_limit=clahe_clip)
        
        # Gabor
        if gabor_strength > 0:
            kernel = cv2.getGaborKernel((15, 15), 3, 0, 10, 0.5, 0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(out, cv2.CV_32F, kernel)
            out = out + gabor_strength * resp
        
        # Unsharp
        blur = ndimage.gaussian_filter(out, sigma=unsharp_sigma)
        out = np.clip(out + unsharp_strength * (out - blur), 0, 1)
        
        return (out * 255).astype(np.uint8)

class ImageQualityMetrics:
    """Mask-aware quality metrics matching the notebook's implementation."""
    @staticmethod
    def _prep(img):
        return img.astype(np.float32) / 255.0 if img.max() > 1.0 else img

    @staticmethod
    def calculate_entropy(img, mask):
        img = ImageQualityMetrics._prep(img)
        if mask.sum() == 0: return 0.0
        hist, _ = np.histogram(img[mask], bins=256, range=(0, 1))
        if hist.sum() == 0: return 0.0
        probs = hist[hist > 0] / hist.sum()
        return -np.sum(probs * np.log2(probs))

    @staticmethod
    def calculate_edge_energy(img, mask):
        img = ImageQualityMetrics._prep(img)
        if mask.sum() == 0: return 0.0
        sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        return np.mean(mag[mask])
    
    @staticmethod
    def calculate_local_contrast(img, mask):
        img = ImageQualityMetrics._prep(img)
        if mask.sum() == 0: return 0.0
        k = np.ones((9, 9), dtype=np.float32) / 81.0
        mu = cv2.filter2D(img, -1, k)
        var = cv2.filter2D(img**2, -1, k) - mu**2
        std = np.sqrt(np.clip(var, 0, None))
        return np.mean(std[mask])

# =====================================================================
# 2. GREY WOLF OPTIMIZER (Matching Notebook)
# =====================================================================

class GreyWolfOptimizer:
    """Grey Wolf Optimization algorithm for parameter optimisation."""
    def __init__(self, objective_func, bounds, num_wolves, max_iterations, convergence_threshold=1e-6):
        self.objective_func = objective_func
        self.bounds = np.array(bounds, dtype=np.float32)
        self.dim = len(bounds)
        self.num_wolves = num_wolves
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.wolves = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.num_wolves, self.dim))
        self.alpha_pos, self.beta_pos, self.delta_pos = np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)
        self.alpha_score, self.beta_score, self.delta_score = -np.inf, -np.inf, -np.inf
        self.convergence_curve = []

    def optimize(self, verbose=False):
        for it in range(self.max_iterations):
            fitness = np.array([self.objective_func(w) for w in self.wolves])
            
            for i in range(self.num_wolves):
                if fitness[i] > self.alpha_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos
                    self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos
                    self.alpha_score, self.alpha_pos = fitness[i], self.wolves[i]
                elif fitness[i] > self.beta_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos
                    self.beta_score, self.beta_pos = fitness[i], self.wolves[i]
                elif fitness[i] > self.delta_score:
                    self.delta_score, self.delta_pos = fitness[i], self.wolves[i]

            a = 2.0 - 2.0 * it / self.max_iterations
            for i in range(self.num_wolves):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                X1 = self.alpha_pos - A1 * abs(C1 * self.alpha_pos - self.wolves[i])
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                X2 = self.beta_pos - A2 * abs(C2 * self.beta_pos - self.wolves[i])
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                X3 = self.delta_pos - A3 * abs(C3 * self.delta_pos - self.wolves[i])
                
                self.wolves[i] = (X1 + X2 + X3) / 3.0
            
            self.wolves = np.clip(self.wolves, self.bounds[:, 0], self.bounds[:, 1])
            self.convergence_curve.append(self.alpha_score)
        return self.alpha_pos, self.alpha_score

# =====================================================================
# 3. ENHANCEMENT PIPELINE FUNCTIONS
# =====================================================================

def create_fitness_function(orig_img, brain_mask, method='adaptive'):
    """Creates the objective function for GWO to maximize."""
    def fitness(params):
        try:
            enhanced_img_uint8 = ImageEnhancer.adaptive_enhancement(orig_img, *params)
            enhanced_img_uint8[~brain_mask] = 0
            
            ent = ImageQualityMetrics.calculate_entropy(enhanced_img_uint8, brain_mask)
            edge = ImageQualityMetrics.calculate_edge_energy(enhanced_img_uint8, brain_mask)
            ctr = ImageQualityMetrics.calculate_local_contrast(enhanced_img_uint8, brain_mask)
            
            return 0.4 * ent + 0.4 * edge * 100 + 0.2 * ctr * 100
        except Exception:
            return -np.inf
    return fitness

def enhance_single_image(task):
    """Full pipeline for one image: load, GWO, apply, save."""
    in_path, out_path, method, gwo_iters, num_wolves = task
    try:
        if out_path.exists():
            return {"success": True, "status": "skipped"}
            
        img_uint8 = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
        if img_uint8 is None: raise IOError(f"Could not load image: {in_path}")
        
        img_float = img_uint8.astype(np.float32) / 255.0
        mask = utils.get_brain_mask(img_float)
        
        # Define bounds for the 'adaptive' method
        bounds = [(1, 4), (0, 0.5), (0.5, 2), (1, 3)]
        
        fit_func = create_fitness_function(img_uint8, mask, method)
        gwo = GreyWolfOptimizer(fit_func, bounds, num_wolves, gwo_iters)
        best_params, _ = gwo.optimize(verbose=False)
        
        enhanced_uint8 = ImageEnhancer.adaptive_enhancement(img_uint8, *best_params)
        enhanced_uint8[~mask] = 0 # Ensure background is black
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), enhanced_uint8)
        return {"success": True, "status": "processed"}
    except Exception as e:
        return {"success": False, "error": str(e), "input_path": str(in_path)}

def enhance_dataset(logger):
    """Processes the entire cropped dataset using GWO-based enhancement."""
    logger.info("\n🧠 STEP 8: GWO IMAGE ENHANCEMENT")
    logger.info("=" * 80)
    
    input_root = config.STEP7_CROPPED_DIR
    output_root = config.STEP8_ENHANCED_DIR
    method = config.GWO_METHOD
    
    if not input_root.exists():
        logger.error(f"❌ Input directory not found: {input_root}. Aborting.")
        return

    tasks = [
        (in_path, output_root / in_path.relative_to(input_root),
         method, config.GWO_ITERATIONS, config.GWO_NUM_WOLVES)
        for in_path in input_root.rglob("*.png")
    ]

    if not tasks:
        logger.error(f"❌ No PNG files found in {input_root} to enhance.")
        return
        
    logger.info(f"🔍 Found {len(tasks)} images to enhance using '{method}' method.")
    
    results = []
    with ThreadPoolExecutor(max_workers=config.MAX_THREADS) as executor:
        futures = {executor.submit(enhance_single_image, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Enhancing ({method})"):
            results.append(future.result())

    success_count = sum(1 for r in results if r["success"] and r["status"] == "processed")
    skipped_count = sum(1 for r in results if r["success"] and r["status"] == "skipped")
    error_count = sum(1 for r in results if not r["success"])
    
    logger.info("\n" + "="*80)
    logger.info("✅ ENHANCEMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"  - Successfully processed: {success_count}")
    logger.info(f"  - Skipped (already exist): {skipped_count}")
    logger.info(f"  - Files with errors: {error_count}")
    
    if error_count > 0:
        logger.warning("Errors occurred during enhancement. First 5 errors:")
        error_msgs = [f"{r['input_path']}: {r['error']}" for r in results if not r['success']]
        for i, msg in enumerate(error_msgs[:5]):
            logger.warning(f"    {i+1}. {msg}")
    
    logger.info(f"  - Output located at: {output_root}")

if __name__ == '__main__':
    from utils.logging_utils import setup_logging
    
    main_logger = setup_logging("image_enhancement_standalone", config.LOG_DIR)
    enhance_dataset(logger=main_logger)
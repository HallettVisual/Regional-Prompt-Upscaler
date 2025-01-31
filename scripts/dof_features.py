import cv2
import numpy as np
import re
import logging
from typing import Set, List, Dict, Optional, Tuple

class DepthOfFieldFeatures:
    """
    Enhanced depth of field and blur analysis functionality for the Regional Prompt Upscaler.
    Uses multiple algorithms for more accurate depth estimation and blur detection.
    """
    
    def __init__(self):
        # Configure logging
        self.logger = logging.getLogger('DepthOfField')
        self.logger.setLevel(logging.DEBUG)
        
        # Default threshold
        self.DEPTH_THRESHOLD = 0.5
        
        # Color words to preserve for context
        self.color_words: Set[str] = {
            'red','blue','green','yellow','white','black','brown',
            'purple','violet','orange','pink','gray','grey',
            'silver','gold','teal','cyan','magenta'
        }

        # Expanded blur and DoF related keywords
        self.dof_keywords: Set[str] = {
            'depth','focal','background','foreground','out','of','field',
            'dof','distance'
        }
        
        # Enhanced depth indicators
        self.depth_terms: Set[str] = {
            'foreground', 'background', 'distance', 'near', 'far',
            'front', 'back', 'close', 'distant', 'depth',
            'proximate', 'remote', 'adjacent', 'immediate'
        }

    def estimate_depth_map(self, image) -> np.ndarray:
        """
        Enhanced depth map estimation using multiple techniques.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            np.ndarray: Enhanced depth map (0-1 range)
        """
        try:
            arr = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            
            # Method 1: Edge-based depth estimation
            edges = cv2.Canny(gray, self.EDGE_THRESHOLD_MIN, self.EDGE_THRESHOLD_MAX)
            edges = 255 - edges
            depth_edges = cv2.distanceTransform(edges, cv2.DIST_L2, 5)
            
            # Method 2: Gradient-based depth estimation
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            depth_gradient = 1 - cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
            
            # Method 3: Intensity-based depth estimation
            depth_intensity = cv2.normalize(gray.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)
            
            # Combine methods with weights
            depth_combined = (0.4 * cv2.normalize(depth_edges, None, 0, 1, cv2.NORM_MINMAX) +
                            0.4 * depth_gradient +
                            0.2 * depth_intensity)
            
            # Apply bilateral filter for edge preservation
            depth_map = cv2.bilateralFilter(depth_combined, 9, 75, 75)
            
            self.logger.debug(f"Depth map generated successfully. Shape: {depth_map.shape}")
            return depth_map
            
        except Exception as e:
            self.logger.error(f"Error in depth map estimation: {str(e)}")
            return np.zeros((image.size[1], image.size[0]), dtype=np.float64)

    def analyze_dof_context(self, prompt: str, context_keywords: set, global_keywords: set,
                           depth_value: Optional[float] = None) -> str:
        """
        Enhanced depth of field analysis with improved context handling.
        
        Args:
            prompt: The input prompt text
            context_keywords: Keywords from surrounding context
            global_keywords: Keywords from global image analysis
            depth_value: Optional normalized depth value for the region (0-1)
            
        Returns:
            str: Modified prompt with enhanced DoF context
        """
        try:
            lower_p = prompt.lower()
            
            # Enhanced DoF detection
            has_dof_prompt = any(kw in lower_p for kw in self.dof_keywords)
            has_dof_context = any(kw in context_keywords for kw in self.dof_keywords)
            has_dof_global = any(kw in global_keywords for kw in self.dof_keywords)
            
            if not (has_dof_prompt or has_dof_context or has_dof_global):
                return prompt
                
            # Enhanced context analysis
            context_colors = context_keywords.intersection(self.color_words)
            global_colors = global_keywords.intersection(self.color_words)
            
            dof_terms = set()
            
            # Enhanced depth-based terms
            if depth_value is not None:
                self.logger.debug(f"Applying depth-based terms with depth value: {depth_value:.2f}")
                if depth_value < 0.3:
                    dof_terms.update(["foreground focus", "sharp details"])
                elif depth_value > 0.7:
                    dof_terms.update(["background blur", "distant details"])
                else:
                    dof_terms.update(["balanced focus", "mid-range clarity"])
            
            # Enhance with color context
            colors = context_colors.union(global_colors)
            if colors:
                dof_terms.update(colors)
                
            # Add depth-specific modifiers
            if "depth" in context_keywords or "field" in context_keywords:
                if depth_value is not None:
                    if depth_value < 0.3:
                        dof_terms.add("shallow depth of field with crisp details")
                    elif depth_value > 0.7:
                        dof_terms.add("deep depth of field with natural blur")
                    else:
                        dof_terms.add("balanced depth of field")
                        
            if not dof_terms:
                return prompt
                
            return " ".join(dof_terms)
            
        except Exception as e:
            self.logger.error(f"Error in DoF context analysis: {str(e)}")
            return prompt

    def analyze_and_apply(self, image):
        """
        Analyzes image depth and applies DoF effects appropriately.
        
        Args:
            image: PIL Image to process
            
        Returns:
            PIL Image: Processed image with DoF effects applied
        """
        try:
            # First analyze the depth
            depth_map = self.estimate_depth_map(image)
            blur_map = self.calculate_blur_map(image)
            
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Apply selective sharpening/blurring based on depth
            depth_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
            
            # Create sharpened and blurred versions
            sharpened = cv2.filter2D(img_array, -1, np.array([[-1,-1,-1],
                                                             [-1, 9,-1],
                                                             [-1,-1,-1]]))
            blurred = cv2.GaussianBlur(img_array, (5,5), 0)
            
            # Blend based on depth
            result = np.zeros_like(img_array)
            for c in range(3):  # Process each color channel
                result[:,:,c] = depth_normalized * sharpened[:,:,c] + \
                               (1 - depth_normalized) * blurred[:,:,c]
            
            # Convert back to PIL
            from PIL import Image
            processed_image = Image.fromarray(result.astype('uint8'))
            
            self.logger.debug("DoF effects applied successfully")
            return processed_image
            
        except Exception as e:
            self.logger.error(f"Error in DoF application: {str(e)}")
            return image
            
    def analyze_region_depth(self, image) -> Tuple[float, Dict[str, float]]:
        """
        Enhanced depth analysis with multiple metrics.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Tuple[float, Dict[str, float]]: Enhanced depth metrics
        """
        try:
            depth_map = self.estimate_depth_map(image)
            
            # Calculate comprehensive statistics
            stats = {
                'avg_depth': float(np.mean(depth_map)),
                'depth_variance': float(np.var(depth_map)),
                'depth_gradient': float(np.gradient(depth_map).max())
            }
            
            self.logger.debug(f"Depth analysis complete. Combined score: {stats['combined_score']:.2f}")
            return stats['combined_score'], stats
            
        except Exception as e:
            self.logger.error(f"Error in region depth analysis: {str(e)}")
            return 0.5, {'avg_depth': 0.5, 'avg_blur': 0.5, 'combined_score': 0.5}

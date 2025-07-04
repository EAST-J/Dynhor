import torch
import torchvision.transforms as transforms

class Dinov2:
	def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=518, half_precision=False, source='github', device="cuda"):
		self.repo_name = repo_name
		self.model_name = model_name
		self.smaller_edge_size = smaller_edge_size
		self.half_precision = half_precision
		self.device = device
		if self.half_precision:
			self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name, source=source).half().to(self.device)
		else:
			self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name, source=source).to(self.device)
		self.model.eval()
		self.feat_size = self.smaller_edge_size // self.model.patch_size
		self.transform_normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  
	def extract_features(self, image_tensor):
		image_tensor = self.transform_normalize(image_tensor)
		tokens = self.model.get_intermediate_layers(image_tensor)[0]
		return tokens
  
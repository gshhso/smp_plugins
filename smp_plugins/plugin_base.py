import torch
from torch import nn
from typing import List

# 插件基类
class PluginBase(nn.Module):
	"""
	插件基类，定义插件接口。
	
	所有插件必须继承此基类并实现bottleneck_layer方法。
	"""
	
	def bottleneck_layer(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
		"""
		在编码器和解码器之间处理特征。
		
		Args:
			features: 编码器输出的特征列表
			
		Returns:
			处理后的特征列表
		"""
		return features
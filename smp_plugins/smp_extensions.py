import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch import Tensor
from typing import Any, Optional, Type, cast, Dict, List, Protocol
from segmentation_models_pytorch.base.model import SegmentationModel

import functools
import inspect

# 导入实际的FarSegPlugin实现
from .fs import FarSegPlugin
from .plugin_base import PluginBase
# 保存对原始create_model的引用
original_create_model = smp.create_model


	

class ExtendableSegmentationModel(nn.Module):
	"""
	可扩展的分割模型，在编码器和解码器之间提供插件支持。

	包装原始SMP模型，并在编码器和解码器之间添加瓶颈层插件。
	"""
	
	def __init__(self, base_model: SegmentationModel, plugins: List[PluginBase] | None = None):
		"""
		初始化可扩展分割模型。
		
		Args:
			base_model: 原始SMP分割模型，应为SegmentationModel类型
			plugins: 插件列表
		"""
		super().__init__()
		# 确保base_model是SegmentationModel类型
		self.base_model = cast(SegmentationModel, base_model)
		if plugins is None:
			plugins = []
		self.plugins = nn.ModuleList(plugins)
		
		# 保存原始模型的组件
		self.encoder = cast(nn.Module, self.base_model.encoder)
		self.decoder = cast(nn.Module, self.base_model.decoder)
		self.segmentation_head = cast(nn.Module, self.base_model.segmentation_head)
		
		
		self.classification_head = getattr(self.base_model, 'classification_head', None)

	def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
		"""
		前向传播。
		
		Args:
			x: 输入张量
			
		Returns:
			模型输出，可能是单个张量或(masks, labels)元组
		"""
		
		# 1. 运行编码器
		features = self.encoder(x)  
  
		# 2. 运行所有插件的瓶颈层处理
		first_feature = features[0] #目前所有smp的decoder都不处理第一个feature
		for plugin in self.plugins:
			plugin = cast(PluginBase, plugin)
			features = plugin.bottleneck_layer(features[1:])
		features = [first_feature] + features
		
		# 3. 运行解码器
		decoder_output = self.decoder(features)  
		
		# 4. 运行分割头
		masks = self.segmentation_head(decoder_output)  
		
		# 5. 处理分类头（如果有）
		if hasattr(self.base_model, 'classification_head') and self.classification_head is not None:
			labels = self.classification_head(features[-1])  
			return masks, labels
			
		return masks
	

# 插件注册表
class PluginRegistry:
	"""
	插件注册表，用于管理和创建插件实例。
	"""
	
	def __init__(self):
		self._registry: Dict[str, Type] = {}  # 使用通用的Type而不是特定的Type[PluginBase]
	
	def register(self, name: str, plugin_class: Type) -> None:
		"""
		注册插件类。
		
		Args:
			name: 插件名称
			plugin_class: 插件类
		"""
		self._registry[name] = plugin_class
	
	def get(self, name: str) -> Type:
		"""
		获取插件类。
		
		Args:
			name: 插件名称
			
		Returns:
			插件类
			
		Raises:
			ValueError: 当插件不存在时
		"""
		if name not in self._registry:
			raise ValueError(f"未知插件: {name}。可用插件: {list(self._registry.keys())}")
		return self._registry[name]
	
	def create(self, 
			  name: str, 
			  encoder: nn.Module, 
			  in_channels: int, 
			  **kwargs: Any) -> Any:  # 返回Any而不是PluginBase
		"""
		创建插件实例。
		
		Args:
			name: 插件名称
			encoder: 编码器实例
			in_channels: 输入通道数
			**kwargs: 传递给插件构造函数的参数
			
		Returns:
			创建的插件实例
			
		Raises:
			ValueError: 当插件不存在或无法获取必要参数时
		"""
		plugin_class = self.get(name)
		
		encoder_channels = self._get_encoder_channels(encoder, in_channels)
		kwargs['feature_channels'] = encoder_channels[1: ]  # remove first skip with same spatial resolution
		
		return plugin_class(**kwargs)
	
	def keys(self) -> List[str]:
		"""
		获取所有已注册的插件名称。
		
		Returns:
			插件名称列表
		"""
		return list(self._registry.keys())
	
	@staticmethod
	def _get_encoder_channels(encoder: nn.Module, in_channels: int) -> List[int]:
		"""
		从编码器获取通道数信息。
		
		Args:
			encoder: 编码器实例
			in_channels: 输入通道数
			
		Returns:
			通道数列表，如果无法获取则返回None
		"""
		# 尝试从编码器属性获取
		if hasattr(encoder, 'out_channels'):
			out_channels = getattr(encoder, 'out_channels')
			if isinstance(out_channels, (list, tuple)) and all(isinstance(x, int) for x in out_channels):
				return list(out_channels[1:])  # 排除第一个特征
		
		if hasattr(encoder, '_out_channels'):
			out_channels = getattr(encoder, '_out_channels')
			if isinstance(out_channels, (list, tuple)) and all(isinstance(x, int) for x in out_channels):
				return list(out_channels[1:])  # 排除第一个特征
		
		raise ValueError(f"无法从编码器'{encoder.__class__.__name__}'获取通道数信息")
		

# 创建全局插件注册表
PLUGIN_REGISTRY = PluginRegistry()

# 注册FarSegPlugin
PLUGIN_REGISTRY.register('FarSegPlugin', FarSegPlugin)

def extended_create_model(
	arch: str,
	encoder_name: str = "resnet34",
	encoder_weights: Optional[str] = "imagenet",
	in_channels: int = 3,
	classes: int = 1,
	plugin_configs: List[tuple[str, Dict[str, Any]]] | None = None,
	**kwargs: Any
) -> nn.Module:
	"""
	扩展的create_model函数，支持插件配置。
	
	Args:
		arch: 架构名称
		encoder_name: 编码器名称
		encoder_weights: 编码器预训练权重
		in_channels: 输入通道数
		classes: 类别数
		plugin_configs: 插件配置列表，每个元素为(插件名, 配置字典)的元组
		**kwargs: 传递给原始create_model的额外参数
		
	Returns:
		可扩展的分割模型
		
	Raises:
		ValueError: 当插件名不在注册表中或无法获取必要参数时
	"""
	# 使用原始函数创建模型
	base_model = original_create_model(
		arch=arch,
		encoder_name=encoder_name,
		encoder_weights=encoder_weights,
		in_channels=in_channels,
		classes=classes,
		**kwargs
	)
	base_model = cast(SegmentationModel, base_model)
 
	# 如果没有插件配置，直接返回原始模型
	if not plugin_configs:
		return base_model
	
	# 构建插件实例
	plugins = []
	encoder_module = cast(nn.Module, base_model.encoder)
	
	for plugin_name, plugin_kwargs in plugin_configs:
		# 使用插件注册表创建插件实例
		plugin = PLUGIN_REGISTRY.create(
			plugin_name, 
			encoder_module, 
			in_channels, 
			**plugin_kwargs
		)
		plugins.append(plugin)
	
	# 包装为可扩展模型
	return ExtendableSegmentationModel(base_model, plugins)

# 替换create_model
smp.create_model = extended_create_model


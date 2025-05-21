"""
FarSeg++前景-场景关系模块(FS Relation Module)实现

此模块实现了FarSeg++论文中的前景-场景关系模块，用于增强地理空间目标分割模型的性能。
模块通过建立前景与场景的关系来增强前景特征的判别能力，抑制背景干扰。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Any, override

from .plugin_base import PluginBase


class ScaleAwareProjection(nn.Module):
	"""
	尺度感知投影模块
	
	将不同尺度的特征图投影到统一的特征空间中，考虑了不同尺度的特性
	"""
	
	def __init__(self, in_channels: int, out_channels: int):
		"""
		初始化尺度感知投影模块
		
		Args:
			in_channels: 输入特征通道数
			out_channels: 输出特征通道数
		"""
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		
		# 1x1卷积进行通道压缩
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
	
	def forward(self, x: Tensor) -> Tensor:
		"""
		前向传播
		
		Args:
			x: 输入特征图，形状为[B, in_channels, H, W]
			
		Returns:
			投影后的特征图，形状为[B, out_channels, H, W]
		"""
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x


class SceneEmbedding(nn.Module):
	"""
	场景嵌入模块
	
	从骨干网络的高层特征中提取场景级别的特征表示
	"""
	
	def __init__(self, in_channels: int, embed_channels: int):
		"""
		初始化场景嵌入模块
		
		Args:
			in_channels: 输入特征通道数（通常是骨干网络最深层的特征通道数）
			embed_channels: 嵌入空间的维度
		"""
		super().__init__()
		self.in_channels = in_channels
		self.embed_channels = embed_channels
		
		# 场景嵌入投影
		self.projection = nn.Conv2d(in_channels, embed_channels, kernel_size=1)
		
		# 全局平均池化，将特征图转换为向量
		self.pool = nn.AdaptiveAvgPool2d(1)
	
	def forward(self, x: Tensor) -> Tensor:
		"""
		前向传播
		
		Args:
			x: 输入特征图，形状为[B, in_channels, H, W]
			
		Returns:
			场景嵌入向量，形状为[B, embed_channels, 1, 1]
		"""
		# 投影到嵌入空间
		x = self.projection(x)
		
		# 全局池化
		x = self.pool(x)
		
		return x


class FeatureEncoder(nn.Module):
	"""
	特征重编码模块
	
	对输入特征进行重编码，为后续的特征增强做准备
	"""
	
	def __init__(self, in_channels: int, out_channels: int):
		"""
		初始化特征重编码模块
		
		Args:
			in_channels: 输入特征通道数
			out_channels: 输出特征通道数
		"""
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
	
	def forward(self, x: Tensor) -> Tensor:
		"""
		前向传播
		
		Args:
			x: 输入特征图，形状为[B, in_channels, H, W]
			
		Returns:
			重编码后的特征图，形状为[B, out_channels, H, W]
		"""
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x


class RelationComputation(nn.Module):
	"""
	前景-场景关系计算模块
	
	计算特征图中的每个空间位置与场景嵌入之间的关系
	"""
	
	def __init__(self):
		"""
		初始化关系计算模块
		"""
		super().__init__()
	
	def forward(self, feature_map: Tensor, scene_embed: Tensor) -> Tensor:
		"""
		前向传播，计算特征图与场景嵌入的关系
		
		Args:
			feature_map: 投影后的特征图，形状为[B, C, H, W]
			scene_embed: 场景嵌入向量，形状为[B, C, 1, 1]
			
		Returns:
			关系图，形状为[B, 1, H, W]
		"""
		# 将scene_embed扩展为与feature_map相同的空间维度
		batch_size, channels = feature_map.shape[:2]
		height, width = feature_map.shape[2:]
		
		# 重塑特征图为[B, C, H*W]
		feat_flat = feature_map.view(batch_size, channels, -1)
		
		# 重塑场景嵌入为[B, C, 1]
		scene_flat = scene_embed.view(batch_size, channels, 1)
		
		# 计算点积相似度: [B, H*W]
		relation = torch.bmm(feat_flat.transpose(1, 2), scene_flat).squeeze(2)
		
		# 重塑为[B, 1, H, W]
		relation = relation.view(batch_size, 1, height, width)
		
		return relation


class ContentPreservation(nn.Module):
	"""
	内容保留机制
	
	FarSeg++中的改进，通过残差连接保留原始特征信息，
	防止特征增强过程中的梯度消失和特征稀疏化
	"""
	
	def __init__(self, lambda_value: float = 0.5):
		"""
		初始化内容保留模块
		
		Args:
			lambda_value: 残差连接的权重系数
		"""
		super().__init__()
		self.lambda_value = lambda_value
	
	def forward(self, enhanced_feat: Tensor, original_feat: Tensor) -> Tensor:
		"""
		前向传播，应用内容保留机制
		
		Args:
			enhanced_feat: 增强后的特征图，形状为[B, C, H, W]
			original_feat: 原始特征图，形状为[B, C, H, W]
			
		Returns:
			应用内容保留后的特征图，形状为[B, C, H, W]
		"""
		return self.lambda_value * enhanced_feat + (1 - self.lambda_value) * original_feat


class FSRelationModule(nn.Module):
	"""
	前景-场景关系模块
	
	FarSeg++的核心组件，通过建立前景与场景的关系来增强前景特征的判别能力。
	"""
	
	def __init__(
		self, 
		in_channels: list[int],
		shared_channels: int = 256, # C_eb
		scene_channels: int = 2048, # C_-1
		lambda_value: float = 0.5
	):
		"""
		初始化前景-场景关系模块
		
		Args:
			in_channels: 各特征层的输入通道数列表，通常对应骨干网络各阶段输出
			shared_channels: 共享特征空间的通道数
			scene_channels: 场景特征通道数，通常是骨干网络最深层的特征通道数
			content_preservation: 是否使用内容保留机制
			lambda_value: 内容保留机制的权重系数
		"""
		super().__init__()
		self.in_channels = in_channels
		self.shared_channels = shared_channels
		self.scene_channels = scene_channels
		
		# 为每个输入特征层创建尺度感知投影
		self.projections = nn.ModuleList([
			ScaleAwareProjection(channels, shared_channels)
			for channels in in_channels
		])
		
		# 为每个输入特征层创建特征重编码
		self.encoders = nn.ModuleList([
			FeatureEncoder(channels, channels)
			for channels in in_channels
		])
		
		# 场景嵌入模块
		self.scene_embedding = SceneEmbedding(scene_channels, shared_channels)
		
		# 关系计算模块
		self.relation_computation = RelationComputation()
		
		# 内容保留模块
		self.content_preservation = ContentPreservation(lambda_value)
	
	def forward(self, features: list[Tensor]) -> list[Tensor]:
		"""
		前向传播
		
		Args:
			features: 特征金字塔特征列表，每个特征形状为[B, C_i, H_i, W_i]
			scene_feature: 场景特征，形状为[B, scene_channels, H_s, W_s]
			
		Returns:
			增强后的特征列表，每个特征形状与输入对应
		"""
		# 提取场景嵌入
		scene_feature = features[-1] #C6(B, C_-1, H_-1, W_-1)
		scene_embed = self.scene_embedding.forward(scene_feature) #C6 -> (B, C_eb, 1, 1)
		
		enhanced_features = []
		
		# 检查特征列表长度是否与投影器数量匹配
		if len(features) != len(self.projections):
			raise ValueError(f"特征列表长度 {len(features)} 与投影器数量 {len(self.projections)} 不匹配")
		
		# 处理每个特征层
		for i, feature in enumerate(features):
			# 投影到共享特征空间
			projected_feat = self.projections[i](feature) #C_i -> (B, C_eb, H_i, W_i)
			
			# 计算关系
			relation = self.relation_computation(projected_feat, scene_embed) #(B, C_eb, H_i, W_i) * (B, C_eb, 1, 1) -> (B, 1, H_i, W_i)
			
			# 归一化关系（使用Sigmoid函数）
			relation = torch.sigmoid(relation)
			
			# 重编码原始特征
			encoded_feat = self.encoders[i](feature) #(B, C_i, H_i, W_i)
			
			# 使用关系图重加权特征
			enhanced_feat = encoded_feat * relation #(B, C_i, H_i, W_i) * (B, 1, H_i, W_i) -> (B, C_i, H_i, W_i)
			
			# 应用内容保留机制（如果启用）
			enhanced_feat = self.content_preservation(enhanced_feat, encoded_feat)
			
			enhanced_features.append(enhanced_feat)
		
		return enhanced_features


class FarSegPlugin(PluginBase):
	"""
	FarSeg++插件
	
	用于SMP扩展的插件，集成了FarSeg++的前景-场景关系模块
	"""
	
	def __init__(
		self,
		feature_channels: list[int],
		pyramid_channels: int = 256,
		lambda_value: float = 0.5,
	):
		"""
		初始化FarSeg++插件
		
		Args:
			pyramid_channels: 特征金字塔通道数
			lambda_value: 内容保留机制的权重系数
			feature_channels: 各尺度特征通道数，必须从编码器获取
		"""
		super().__init__()
		self.pyramid_channels = pyramid_channels
		self.lambda_value = lambda_value
		
		self.feature_channels = feature_channels
		
		# 在初始化时创建fs_relation模块
		self.fs_relation = FSRelationModule(
			in_channels=self.feature_channels,
			shared_channels=self.pyramid_channels,
			scene_channels=self.feature_channels[-1],
			lambda_value=self.lambda_value
		)
	
	@override
	def bottleneck_layer(self, features: list[Tensor]) -> list[Tensor]:
		"""
		瓶颈层处理，适配SMP扩展接口
		
		Args:
			features: 编码器输出的特征列表
			
		Returns:
			应用前景-场景关系增强后的特征列表
		"""
		# 获取除第一个特征外的所有特征
		features_for_fs = features[1:]
		
		# 检查特征列表长度是否与feature_channels匹配
		if len(features_for_fs) != len(self.feature_channels):
			raise ValueError(f"特征列表长度 {len(features_for_fs)} 与feature_channels长度 {len(self.feature_channels)} 不匹配")
		
		# 应用前景-场景关系增强
		enhanced_features = self.fs_relation(features_for_fs)
		
		# 将增强的特征与第一个原始特征合并
		return [features[0]] + enhanced_features
		








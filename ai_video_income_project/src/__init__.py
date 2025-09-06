"""
AI Video Income Generator Project

A comprehensive toolkit for creating passive income through AI-powered video clipping.
This project provides tools to analyze long-form content, extract valuable segments,
and monetize them through YouTube uploads and view-based payments.
"""

__version__ = "1.0.0"
__author__ = "AI Video Income Team"
__description__ = "Generate passive income through AI-powered video clipping"

# Import main components
from .video_processor import VideoAnalyzer, ClipGenerator, ThumbnailCreator
from .ai_agents import ContentAnalyzer, SegmentFinder, TitleGenerator
from .uploader import YouTubeUploader, MetadataManager, Tracking
from .utils import VideoUtils, AIUtils, PaymentUtils

__all__ = [
    "VideoAnalyzer",
    "ClipGenerator", 
    "ThumbnailCreator",
    "ContentAnalyzer",
    "SegmentFinder",
    "TitleGenerator",
    "YouTubeUploader",
    "MetadataManager",
    "Tracking",
    "VideoUtils",
    "AIUtils",
    "PaymentUtils"
] 
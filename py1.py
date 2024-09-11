import torch
from transformers import DalleBartProcessor, DalleBartForConditionalGeneration

processor = DalleBartProcessor.from_pretrained("dalle-bart-123M-mega")
model = DalleBartForConditionalGeneration.from_pretrained("dalle-bart-123M-mega")
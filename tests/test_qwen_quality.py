import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from whisperjav.modules.qwen_asr import QwenASR
from whisperjav.modules.subtitle_pipeline.generators.qwen3 import Qwen3TextGenerator
from whisperjav.pipelines.qwen_pipeline import QwenPipeline
from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
from whisperjav.modules.subtitle_pipeline.types import HardeningConfig, TemporalFrame, TimestampMode

@pytest.fixture
def mock_qwen_asr_module_with_thinker():
    """Mock qwen_asr package with a model that has thinker.generation_config."""
    mock_module = MagicMock()
    
    class MockThinker:
        def __init__(self):
            self.generation_config = MagicMock()
            
    class MockInnerModel:
        def __init__(self):
            self.thinker = MockThinker()
            
    class MockModel:
        def __init__(self):
            self.model = MockInnerModel()
            
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            return cls()
            
    mock_module.Qwen3ASRModel = MockModel
    with patch.dict(sys.modules, {'qwen_asr': mock_module}):
        yield mock_module

def test_qwen_asr_initialization_defaults():
    asr = QwenASR()
    assert asr.num_beams == 1
    assert asr.do_sample is False
    assert asr.temperature == 1.0

def test_qwen_asr_initialization_custom():
    asr = QwenASR(num_beams=4, do_sample=True, temperature=0.7)
    assert asr.num_beams == 4
    assert asr.do_sample is True
    assert asr.temperature == 0.7

def test_qwen_asr_apply_generation_config(mock_qwen_asr_module_with_thinker):
    asr = QwenASR(repetition_penalty=1.2, num_beams=4, do_sample=True, temperature=0.8)
    asr.load_model_text_only()
    
    # Check that gen_config received all our custom attributes
    gen_config = asr.model.model.thinker.generation_config
    assert gen_config.repetition_penalty == 1.2
    assert gen_config.num_beams == 4
    assert gen_config.do_sample is True
    assert gen_config.temperature == 0.8
    assert gen_config.early_stopping is True

def test_qwen3_text_generator_propagation(mock_qwen_asr_module_with_thinker):
    gen = Qwen3TextGenerator(num_beams=3, do_sample=True, temperature=0.9)
    gen.load()
    
    assert gen._asr.num_beams == 3
    assert gen._asr.do_sample is True
    assert gen._asr.temperature == 0.9

def test_qwen_pipeline_parameter_propagation(mock_qwen_asr_module_with_thinker):
    # Mock other dependencies of the pipeline that might hit real files/APIs
    with patch('whisperjav.pipelines.qwen_pipeline.AudioExtractor'), \
         patch('whisperjav.pipelines.qwen_pipeline.SRTStitcher'), \
         patch('whisperjav.pipelines.qwen_pipeline.SRTPostProcessor'):
        
        pipeline = QwenPipeline(
            qwen_num_beams=5,
            qwen_do_sample=True,
            qwen_temperature=0.6,
            qwen_context_propagation=True
        )
        
        # Verify stored in _asr_config
        assert pipeline._asr_config["num_beams"] == 5
        assert pipeline._asr_config["do_sample"] is True
        assert pipeline._asr_config["temperature"] == 0.6
        assert pipeline.context_propagation is True
        
        # Verify passed to DecoupledSubtitlePipeline
        assert pipeline._subtitle_pipeline.context_propagation is True

def test_orchestrator_context_propagation():
    # Setup mock components
    framer = MagicMock()
    generator = MagicMock()
    cleaner = MagicMock()
    aligner = MagicMock()
    
    # We will configure generator.generate_batch to return custom texts
    class MockResult:
        def __init__(self, text):
            self.text = text
            
    generator.generate_batch.side_effect = [
        [MockResult("こんにちは。"), MockResult("私はQwenです。")], # Scene 0
        [MockResult("次のシーンです。")]                        # Scene 1
    ]
    
    # Cleaner passes through or formats
    cleaner.clean_batch.side_effect = lambda x: x
    
    orchestrator = DecoupledSubtitlePipeline(
        framer=framer,
        generator=generator,
        cleaner=cleaner,
        aligner=aligner,
        hardening_config=HardeningConfig(
            timestamp_mode=TimestampMode.VAD_ONLY,
            regroup_mode="off"
        ),
        context="Global Context",
        context_propagation=True
    )
    
    # Mocking frames and paths
    scene_frames = [
        [TemporalFrame(start=0.0, end=2.0), TemporalFrame(start=2.0, end=4.0)],
        [TemporalFrame(start=4.0, end=6.0)]
    ]
    frame_audio_paths = [
        [Path("scene0_f0.wav"), Path("scene0_f1.wav")],
        [Path("scene1_f0.wav")]
    ]
    scene_durations = [4.0, 2.0]
    
    # Run the generator step
    scene_texts = orchestrator._step2_4_generate_and_clean(
        scene_frames=scene_frames,
        frame_audio_paths=frame_audio_paths,
        scene_durations=scene_durations
    )
    
    # Assert scene texts are returned correctly
    assert scene_texts == [["こんにちは。", "私はQwenです。"], ["次のシーンです。"]]
    
    # Verify that generator.generate_batch was called twice
    assert generator.generate_batch.call_count == 2
    
    # Inspect the contexts passed to generate_batch in each call
    calls = generator.generate_batch.call_args_list
    
    # Call 1 (Scene 0) should use the global context
    assert calls[0][1]["contexts"] == ["Global Context", "Global Context"]
    
    # Call 2 (Scene 1) should include the propagated context from Scene 0
    # Clean transcript for Scene 0 is "こんにちは。私はQwenです。"
    # Preceding transcript is the last 200 chars, so it should end with that.
    expected_context = "Global Context\nPrevious: こんにちは。私はQwenです。"
    assert calls[1][1]["contexts"] == [expected_context]

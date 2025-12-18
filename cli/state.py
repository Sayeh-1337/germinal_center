"""Pipeline state management for resume functionality"""
import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

STATE_FILE_NAME = ".pipeline_state.json"


class PipelineState:
    """Manages pipeline execution state for resume functionality"""
    
    STEPS = ['preprocess', 'segment', 'extract', 'analyze']
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.state_file = os.path.join(output_dir, STATE_FILE_NAME)
        self.state = self._load_or_create()
    
    def _load_or_create(self) -> Dict[str, Any]:
        """Load existing state or create new one"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load state file: {e}. Creating new state.")
        
        return self._create_new_state()
    
    def _create_new_state(self) -> Dict[str, Any]:
        """Create a fresh state"""
        return {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'current_step': None,
            'completed_steps': [],
            'steps': {
                step: {
                    'status': 'pending',  # pending, in_progress, completed, failed
                    'started_at': None,
                    'completed_at': None,
                    'processed_files': [],
                    'total_files': 0,
                    'error': None
                } for step in self.STEPS
            }
        }
    
    def save(self):
        """Save current state to file"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.state['updated_at'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def start_step(self, step: str, total_files: int = 0):
        """Mark a step as started"""
        self.state['current_step'] = step
        self.state['steps'][step]['status'] = 'in_progress'
        self.state['steps'][step]['started_at'] = datetime.now().isoformat()
        self.state['steps'][step]['total_files'] = total_files
        self.state['steps'][step]['error'] = None
        self.save()
        logger.info(f"Started step: {step}")
    
    def complete_step(self, step: str):
        """Mark a step as completed"""
        self.state['steps'][step]['status'] = 'completed'
        self.state['steps'][step]['completed_at'] = datetime.now().isoformat()
        if step not in self.state['completed_steps']:
            self.state['completed_steps'].append(step)
        self.state['current_step'] = None
        self.save()
        logger.info(f"Completed step: {step}")
    
    def fail_step(self, step: str, error: str):
        """Mark a step as failed"""
        self.state['steps'][step]['status'] = 'failed'
        self.state['steps'][step]['error'] = error
        self.save()
        logger.error(f"Step {step} failed: {error}")
    
    def mark_file_processed(self, step: str, filename: str):
        """Mark a file as processed in a step"""
        if filename not in self.state['steps'][step]['processed_files']:
            self.state['steps'][step]['processed_files'].append(filename)
            # Save periodically (every 5 files) to avoid too many writes
            if len(self.state['steps'][step]['processed_files']) % 5 == 0:
                self.save()
    
    def is_file_processed(self, step: str, filename: str) -> bool:
        """Check if a file has already been processed in a step"""
        return filename in self.state['steps'][step]['processed_files']
    
    def get_unprocessed_files(self, step: str, all_files: List[str]) -> List[str]:
        """Get list of files that haven't been processed yet"""
        processed = set(self.state['steps'][step]['processed_files'])
        return [f for f in all_files if os.path.basename(f) not in processed]
    
    def is_step_completed(self, step: str) -> bool:
        """Check if a step is completed"""
        return self.state['steps'][step]['status'] == 'completed'
    
    def get_resume_steps(self, requested_steps: List[str]) -> List[str]:
        """Get steps that need to be run when resuming
        
        Returns only steps that are not completed or are in progress
        """
        resume_steps = []
        for step in requested_steps:
            status = self.state['steps'][step]['status']
            if status != 'completed':
                resume_steps.append(step)
        return resume_steps
    
    def get_progress_summary(self) -> str:
        """Get a summary of pipeline progress"""
        lines = ["Pipeline State Summary:"]
        lines.append("-" * 40)
        
        for step in self.STEPS:
            step_state = self.state['steps'][step]
            status = step_state['status']
            processed = len(step_state['processed_files'])
            total = step_state['total_files']
            
            if status == 'completed':
                lines.append(f"  {step}: COMPLETED")
            elif status == 'in_progress':
                if total > 0:
                    lines.append(f"  {step}: IN PROGRESS ({processed}/{total} files)")
                else:
                    lines.append(f"  {step}: IN PROGRESS")
            elif status == 'failed':
                lines.append(f"  {step}: FAILED - {step_state['error']}")
            else:
                lines.append(f"  {step}: pending")
        
        lines.append("-" * 40)
        return "\n".join(lines)
    
    def reset(self):
        """Reset state to start fresh"""
        self.state = self._create_new_state()
        self.save()
        logger.info("Pipeline state reset")
    
    def reset_step(self, step: str):
        """Reset a specific step"""
        self.state['steps'][step] = {
            'status': 'pending',
            'started_at': None,
            'completed_at': None,
            'processed_files': [],
            'total_files': 0,
            'error': None
        }
        if step in self.state['completed_steps']:
            self.state['completed_steps'].remove(step)
        self.save()
        logger.info(f"Step {step} reset")


def get_state(output_dir: str) -> PipelineState:
    """Get or create pipeline state for an output directory"""
    return PipelineState(output_dir)


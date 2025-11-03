"""
Ablation Training Progress Monitor
===================================

Monitors the ablation study training in real-time and alerts when complete.
Tracks:
- Current seed and epoch
- Training time elapsed and estimated time remaining
- Loss convergence
- Performance metrics

This script runs periodically to update a status file.
"""

import json
import re
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AblationMonitor:
    """Monitor ablation training progress"""

    def __init__(self,
                 log_file: Path = Path("/e/Research/zero shot lung/ablation_training.log"),
                 results_dir: Path = Path("/e/Research/zero shot lung/results_full_scale_ablation"),
                 status_file: Path = Path("/e/Research/zero shot lung/results_full_scale_corrected/ABLATION_PROGRESS_REALTIME.txt")):
        """
        Initialize monitor.

        Args:
            log_file: Path to training log
            results_dir: Directory where results will be saved
            status_file: File to write status to
        """
        self.log_file = log_file
        self.results_dir = results_dir
        self.status_file = status_file
        self.start_time = None

    def parse_log_tail(self, n_lines: int = 500) -> dict:
        """
        Parse the last n lines of the training log.

        Args:
            n_lines: Number of lines to read from end

        Returns:
            Dictionary with parsed information
        """
        if not self.log_file.exists():
            return {'status': 'not_started', 'message': 'Log file not found'}

        # Read last n lines
        try:
            with open(self.log_file, 'r', errors='ignore') as f:
                lines = f.readlines()

            if not lines:
                return {'status': 'not_started', 'message': 'Log file empty'}

            recent = lines[-n_lines:]
            log_text = ''.join(recent)

        except Exception as e:
            logger.error(f"Error reading log: {e}")
            return {'status': 'error', 'message': str(e)}

        # Parse for current state
        result = {
            'status': 'training',
            'timestamp': datetime.now().isoformat()
        }

        # Check for seed and epoch
        seed_match = re.search(r'SEED (\d+)', log_text)
        epoch_match = re.search(r'Epoch (\d+)/40', log_text)
        batch_match = re.search(r'Training:\s+(\d+)%.*?(\d+)/6250', log_text)

        if seed_match:
            result['current_seed'] = int(seed_match.group(1))
        if epoch_match:
            result['current_epoch'] = int(epoch_match.group(1))
        if batch_match:
            result['epoch_progress_pct'] = int(batch_match.group(1))
            result['current_batch'] = int(batch_match.group(2))

        # Check for completion markers
        if 'ALL SEEDS COMPLETED' in log_text:
            result['status'] = 'completed_all'
        elif 'SEED' in log_text and 'completed' in log_text.lower():
            if 'SEED 456' in log_text:
                result['status'] = 'completed_all'
            else:
                result['status'] = 'between_seeds'

        # Parse loss values
        loss_matches = re.findall(r'loss=([0-9.]+)', log_text[-1000:])
        if loss_matches:
            result['latest_loss'] = float(loss_matches[-1])
            result['loss_history'] = [float(x) for x in loss_matches[-10:]]

        # Check for errors
        if 'ERROR' in log_text or 'Exception' in log_text:
            result['status'] = 'error'
            error_match = re.search(r'ERROR[:\s]*(.+)', log_text)
            if error_match:
                result['error_message'] = error_match.group(1)[:100]

        return result

    def check_results_saved(self) -> bool:
        """Check if results have been saved to disk"""
        summary_file = self.results_dir / "summary_results.json"
        return summary_file.exists()

    def estimate_completion_time(self, status: dict) -> str:
        """
        Estimate when ablation will complete.

        Args:
            status: Current status dictionary

        Returns:
            Estimated completion time string
        """
        # Baseline: ~11.3 hours per seed
        # 3 seeds total = ~33 hours
        # Started at 2025-11-03 02:13 UTC

        started = datetime(2025, 11, 3, 2, 13)
        expected_hours = 33

        if status.get('current_seed'):
            # Rough estimate based on seed number
            seed_num = status['current_seed']
            if seed_num == 42:
                seed_progress = 0.0
            elif seed_num == 123:
                seed_progress = 1.0
            elif seed_num == 456:
                seed_progress = 2.0
            else:
                seed_progress = 0.0

            # Epoch progress (0-40 per seed)
            epoch_progress = status.get('current_epoch', 1) / 40.0

            # Total progress
            total_progress = (seed_progress + epoch_progress) / 3.0
            elapsed = (datetime.now() - started).total_seconds() / 3600  # hours

            if elapsed > 0 and total_progress > 0:
                estimated_total = elapsed / total_progress
                completion_time = started + timedelta(hours=estimated_total)
                hours_remaining = estimated_total - elapsed
                return f"{completion_time.strftime('%2025-11-%d %H:%M UTC')} (~{hours_remaining:.1f} hours remaining)"

        completion_time = started + timedelta(hours=expected_hours)
        return completion_time.strftime('%2025-11-%d %H:%M UTC')

    def write_status(self, status: dict) -> None:
        """
        Write current status to file.

        Args:
            status: Status dictionary
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ABLATION STUDY - REAL-TIME PROGRESS MONITOR")
        lines.append(f"Updated: {datetime.now().strftime('%2025-11-%d %H:%M:%S UTC')}")
        lines.append("=" * 80)
        lines.append("")

        # Status summary
        status_map = {
            'not_started': 'WAITING TO START',
            'training': 'TRAINING IN PROGRESS',
            'between_seeds': 'BETWEEN SEEDS',
            'completed_all': 'COMPLETED',
            'error': 'ERROR DETECTED'
        }
        status_str = status_map.get(status.get('status'), 'UNKNOWN')
        lines.append(f"Overall Status: {status_str}")
        lines.append("")

        # Current progress
        if status.get('current_seed'):
            lines.append("CURRENT PROGRESS:")
            lines.append(f"  Seed: {status['current_seed']}")
            lines.append(f"  Epoch: {status.get('current_epoch', '?')}/40")
            if 'current_batch' in status:
                lines.append(f"  Batch: {status['current_batch']}/6250 ({status.get('epoch_progress_pct', '?')}%)")
            if 'latest_loss' in status:
                lines.append(f"  Training Loss: {status['latest_loss']:.4f}")
            lines.append("")

        # Timing
        lines.append("TIMING ESTIMATE:")
        lines.append(f"  Expected Completion: {self.estimate_completion_time(status)}")
        lines.append("")

        # Results
        lines.append("RESULTS:")
        if self.check_results_saved():
            lines.append("  Summary results file: SAVED")
            try:
                with open(self.results_dir / "summary_results.json") as f:
                    results = json.load(f)
                    lines.append(f"  Number of seeds with results: {len(results.get('all_results', []))}")
            except:
                pass
        else:
            lines.append("  Summary results file: NOT YET SAVED")
        lines.append("")

        # Error handling
        if status.get('status') == 'error':
            lines.append("ERROR DETAILS:")
            lines.append(f"  {status.get('error_message', 'Unknown error')}")
            lines.append("")

        # Instructions
        lines.append("NEXT STEPS (UPON COMPLETION):")
        lines.append("  1. Verify all 3 seed results are saved")
        lines.append("  2. Run: python delong_statistical_analysis.py")
        lines.append("  3. Check results in delong_analysis_report.txt")
        lines.append("  4. Update LaTeX macros with ΔAUC and p-values")
        lines.append("")

        # Write file
        with open(self.status_file, 'w') as f:
            f.write('\n'.join(lines))

        # Also print to console
        print('\n'.join(lines))

    def run(self) -> None:
        """Execute monitoring cycle"""
        logger.info("Checking ablation progress...")
        status = self.parse_log_tail()
        self.write_status(status)
        logger.info(f"Status: {status.get('status')}")


def main():
    """Main execution"""
    monitor = AblationMonitor()
    monitor.run()


if __name__ == "__main__":
    main()

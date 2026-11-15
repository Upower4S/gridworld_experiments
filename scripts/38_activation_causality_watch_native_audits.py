#!/usr/bin/env python3
"""Watch six fixed native cohorts using filenames only, then run saved-data audits.

--once is a read-only readiness check: it NEVER launches an audit, even when a
cohort is ready. Regular mode runs the pinned archived auditor once per complete
cohort, preserving immutable automated-recomputation reports and logs. These are
not human reviews. No model framework, worker lock or model action is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time


CAMPAIGN = Path('/Users/wws/SPAR_code/outputs/activation_causality/campaign_20260912')
PLAN = 'reviews/subspace_complete_native_plan_v1.json'
AUDITOR = 'reviews/subspace_complete_independent_audit_v1.py'
PINS = {
    PLAN: '4c77834e1dc6cecd7bf10f446b5b4f5ffe8e0123463185773001b6da400a86e0',
    AUDITOR: '96699e034a7eb6e6c1b18ce22d190321c2cbb5d389edebbc2e251a4c053ed6d9',
    'specs/subspace_wall_l0_r1_native_v1.json': '246ccd1001c3413fce23b31537fbf5f5ecd3dc475d8cb34966ce9f0f84c09243',
    'specs/subspace_wall_l0_r4_native_v1.json': '89476a7cdfc566b43c423f459244a85a6eacd4fb350eea956b77db3410512dee',
    'specs/subspace_key_l0_r1_native_v1.json': '0b51408d5517717aa9d60216abc26dce9afb8a9aadb780d91ee43fe92f152a06',
    'specs/subspace_key_l0_r4_native_v1.json': '5ec2158a3a9d045b0b8255b884004b809e5bb0788b78a86ef15848b56d615415',
    'specs/subspace_door_l0_r1_native_v1.json': 'e76566ec601d136c48b24fd108f2a6848d6bf9cf8f32bf355e482aeb3a57649a',
    'specs/subspace_door_l0_r4_native_v1.json': '3d839097f1b27a4ff220f3ec685c475edb1ad3bd6fa74dc2cc312de388a6d267',
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def immutable_json(path: Path, value: dict) -> None:
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def load_contract(campaign: Path, pins: dict = PINS) -> dict:
    """Read only pinned plan/spec/auditor bytes; injectable pins support CPU fixtures."""
    campaign = Path(campaign).resolve()
    contents = {}
    for relative, expected in pins.items():
        raw = (campaign / relative).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, f'Pinned SHA mismatch: {relative}')
        contents[relative] = raw
    plan = json.loads(contents[PLAN])
    identifiers = [fit['native_experiment_id'] for fit in plan['fits']]
    require(len(identifiers) == len(set(identifiers)) == 6, 'Require six distinct planned native cohorts')
    require(set(pins) == {PLAN, AUDITOR, *(f'specs/{eid}.json' for eid in identifiers)}, 'Pinned dependency inventory differs from plan')
    expected_names = {}
    for eid in identifiers:
        require(re.fullmatch(r'[A-Za-z0-9_-]+', eid) is not None, 'Invalid planned experiment ID')
        spec = json.loads(contents[f'specs/{eid}.json'])
        jobs = spec['jobs']
        require(spec['experiment_id'] == eid and len(jobs) == 56, 'Require exact 56-job native specification')
        require(all(job.get('natural') is True and re.fullmatch(r'[0-9a-f]{24}', job.get('job_id', ''))
                    for job in jobs), 'Invalid native job inventory')
        names = {job['job_id'] + '.json' for job in jobs}
        require(len(names) == 56, 'Duplicate planned native job IDs')
        require(not (campaign / 'reviews' / f'{eid}_retired.json').exists(), f'Retired cohort: {eid}')
        expected_names[eid] = names
    return {'campaign': campaign, 'pins': dict(pins), 'expected_names': expected_names,
            'watcher_source_sha256': sha(Path(__file__))}


def readiness(contract: dict) -> list[dict]:
    """List top-level result filenames; no result file contents are opened."""
    result = []
    for eid, expected in contract['expected_names'].items():
        present = {path.name for path in (contract['campaign'] / 'results' / eid).glob('*.json') if path.is_file()}
        missing, extra = sorted(expected - present), sorted(present - expected)
        result.append({'experiment_id': eid, 'expected_files': 56, 'saved_files': len(present),
                       'missing_count': len(missing), 'unexpected_files': extra,
                       'filename_inventory_sha256': hashlib.sha256('\n'.join(sorted(present)).encode()).hexdigest(),
                       'state': 'error' if extra else 'pending' if missing else 'ready'})
    return result


def artifact_paths(output: Path, eid: str) -> tuple[Path, Path, Path]:
    stem = output / f'{eid}.automated_recomputation'
    return tuple(Path(str(stem) + suffix) for suffix in ('.audit.json', '.log', '.json'))


def output_is_separate(campaign: Path, output: Path) -> bool:
    return not any(output.resolve().is_relative_to(campaign.resolve() / name)
                   for name in ('results', 'specs', 'bases', 'source_snapshots'))


def already_completed(output: Path, eid: str, contract: dict) -> bool:
    audit_path, log_path, record_path = artifact_paths(output, eid)
    if not record_path.exists():
        require(not audit_path.exists() and not log_path.exists(), f'Interrupted prior audit artifacts retained: {eid}')
        return False
    record = json.loads(record_path.read_bytes())
    require(record.get('format') == 'automated_subspace_native_audit_recomputation_v1'
            and record.get('status') == 'complete' and record.get('experiment_id') == eid
            and record.get('pinned_sha256') == contract['pins'], f'Prior failed or incompatible audit retained: {eid}')
    require(sha(audit_path) == record['audit_sha256'] and sha(log_path) == record['log_sha256'], f'Prior audit artifacts changed: {eid}')
    audited = json.loads(audit_path.read_bytes())
    manifest = audited.get('input_file_sha256')
    require(isinstance(manifest, dict) and bool(manifest), f'Prior audit lacks its input manifest: {eid}')
    repo = contract['campaign'].parents[2]
    for relative, digest in manifest.items():
        path = (repo / relative).resolve()
        require(path.is_relative_to(repo) and sha(path) == digest, f'Previously audited input changed: {relative}')
    return True


def audit_ready(contract: dict, eid: str, output: Path, timeout: float, runner=subprocess.run) -> dict:
    """Launch only the pinned CPU auditor; retain failure metadata before raising."""
    audit_path, log_path, record_path = artifact_paths(output, eid)
    require(not any(path.exists() for path in (audit_path, log_path, record_path)), f'Audit output already exists: {eid}')
    command = [sys.executable, '-I', '-B', str(contract['campaign'] / AUDITOR), '--experiment', eid, '--output', str(audit_path)]
    record = {'format': 'automated_subspace_native_audit_recomputation_v1', 'experiment_id': eid,
              'started_at': now(), 'status': 'error', 'human_review_completed': False,
              'interpretation': 'Automated saved-data recomputation; separate human interpretation and review remain required.',
              'pinned_sha256': contract['pins'], 'command': command,
              'watcher_source_sha256': contract['watcher_source_sha256'],
              'audit_path': str(audit_path), 'log_path': str(log_path)}
    with log_path.open('xb') as log:
        try:
            fresh = load_contract(contract['campaign'], contract['pins'])
            require(fresh['watcher_source_sha256'] == contract['watcher_source_sha256'], 'Watcher source changed')
            state = next(item for item in readiness(fresh) if item['experiment_id'] == eid)
            require(state['state'] == 'ready', f'Filename inventory changed before audit: {eid}')
            result = runner(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            record['returncode'] = result.returncode
            require(result.returncode == 0, f'Auditor exited {result.returncode}: {eid}')
            fresh = load_contract(contract['campaign'], contract['pins'])
            require(fresh['watcher_source_sha256'] == contract['watcher_source_sha256'], 'Watcher source changed')
            audited = json.loads(audit_path.read_bytes())
            cohort = audited.get('cohort', {})
            require(audited.get('format') == 'subspace_complete_independent_cohort_audit_v1'
                    and audited.get('audit_source_sha256') == contract['pins'][AUDITOR]
                    and cohort.get('experiment_id') == eid and cohort.get('state') == 'complete'
                    and cohort.get('planned_jobs') == cohort.get('saved_file_count') == 56
                    and cohort.get('outcomes_read') is True, f'Auditor did not certify the complete expected cohort: {eid}')
            record['status'] = 'complete'
        except Exception as exc:
            record.update(error_type=type(exc).__name__, error=str(exc))
            log.write(('\nWatcher failure: ' + type(exc).__name__ + ': ' + str(exc) + '\n').encode())
    record.update(finished_at=now(), audit_sha256=sha(audit_path) if audit_path.exists() else None,
                  log_sha256=sha(log_path))
    immutable_json(record_path, record)
    require(record['status'] == 'complete', f'Automated audit failed; preserved {record_path}')
    return record


@contextmanager
def watcher_lock(output: Path):
    """Only protects this CPU watcher's outputs; never touches a model-worker lock."""
    with (output / '.watcher.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def watch(contract: dict, output: Path, *, seconds: float, poll: float, audit_timeout: float,
          runner=subprocess.run, sleeper=time.sleep, monotonic=time.monotonic) -> dict:
    require(all(math.isfinite(value) and value > 0 for value in (seconds, poll, audit_timeout)) and poll <= 30,
            'Positive finite timeouts and poll interval <=30 seconds required')
    require(output_is_separate(contract['campaign'], output),
            'Watcher outputs must be separate from experimental inputs and raw results')
    output.mkdir(parents=True, exist_ok=True)
    stop_at = monotonic() + seconds
    with watcher_lock(output):
        try:
            while True:
                fresh = load_contract(contract['campaign'], contract['pins'])
                require(fresh['watcher_source_sha256'] == contract['watcher_source_sha256'], 'Watcher source changed')
                states = readiness(contract)
                require(not any(item['state'] == 'error' for item in states), 'Unexpected result filenames; no audit launched')
                completed = []
                for item in states:
                    eid = item['experiment_id']
                    if item['state'] != 'ready':
                        require(not any(path.exists() for path in artifact_paths(output, eid)),
                                f'Previously audited filename inventory changed: {eid}')
                        continue
                    if already_completed(output, eid, contract):
                        completed.append(eid)
                        continue
                    remaining = stop_at - monotonic()
                    if remaining <= 0:
                        return {'state': 'timeout', 'automated_completed': completed, 'cohorts': states}
                    if item['state'] == 'ready':
                        audit_ready(contract, eid, output, min(audit_timeout, remaining), runner)
                        completed.append(eid)
                        print(json.dumps({'state': 'automated_audit_complete', 'experiment_id': eid,
                                          'human_review_completed': False}), flush=True)
                if len(completed) == 6:
                    return {'state': 'all_automated_audits_complete', 'automated_completed': completed,
                            'human_review_completed': False}
                remaining = stop_at - monotonic()
                if remaining <= 0:
                    return {'state': 'timeout', 'automated_completed': completed, 'cohorts': states}
                sleeper(min(poll, remaining))
        except Exception as exc:
            path = output / ('watcher_failure_' + str(time.time_ns()) + '.json')
            immutable_json(path, {'format': 'automated_audit_watcher_failure_v1', 'created_at': now(),
                                 'error_type': type(exc).__name__, 'error': str(exc),
                                 'pinned_sha256': contract['pins'], 'model_actions_taken': False})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', help='Filename-only dry run; never launch child audits')
    parser.add_argument('--output-dir', type=Path, default=CAMPAIGN / 'reviews/native_automated_recomputation_v1')
    parser.add_argument('--timeout-seconds', type=float, default=21600)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--audit-timeout-seconds', type=float, default=120)
    args = parser.parse_args()
    entered_watch = False
    output_safe = False
    try:
        args.output_dir = args.output_dir.resolve()
        output_safe = output_is_separate(CAMPAIGN, args.output_dir)
        require(output_safe, 'Watcher outputs must be separate from experimental inputs and raw results')
        require(all(math.isfinite(value) and value > 0 for value in
                    (args.timeout_seconds, args.poll_seconds, args.audit_timeout_seconds)) and args.poll_seconds <= 30,
                'Positive finite timeouts and poll interval <=30 seconds required')
        contract = load_contract(CAMPAIGN)
        if args.once:
            states = readiness(contract)
            print(json.dumps({'state': 'dry_run', 'child_audits_launched': 0, 'result_contents_read': False,
                              'cohorts': states}, indent=2))
            if any(item['state'] == 'error' for item in states):
                raise ValueError('Unexpected result filenames')
            return
        config = json.loads((CAMPAIGN / 'campaign.json').read_bytes())
        deadline = datetime.fromisoformat(config['deadline'])
        require(deadline.tzinfo is not None, 'Campaign deadline must include a timezone')
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        seconds = min(args.timeout_seconds, remaining)
        if seconds <= 0:
            print(json.dumps({'state': 'deadline_reached', 'child_audits_launched': 0}))
            return
        entered_watch = True
        result = watch(contract, args.output_dir.resolve(), seconds=seconds, poll=args.poll_seconds,
                       audit_timeout=args.audit_timeout_seconds)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        if not args.once and not entered_watch and output_safe:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            immutable_json(args.output_dir / ('watcher_startup_failure_' + str(time.time_ns()) + '.json'),
                           {'created_at': now(), 'error_type': type(exc).__name__, 'error': str(exc),
                            'stage': 'before any child audit', 'model_actions_taken': False})
        parser.exit(1, f'Native audit watcher stopped: {exc}\n')


if __name__ == '__main__':
    main()

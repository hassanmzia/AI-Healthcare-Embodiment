"""Main pipeline orchestrator for the MS screening workflow."""
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from django.conf import settings
from django.utils import timezone

from .base import AgentOutput
from .retrieval import RetrievalAgent
from .phenotyping import PhenotypingAgentV2
from .notes_imaging import NotesImagingAgent
from .safety import SafetyGovernanceAgent
from .coordinator import Coordinator

logger = logging.getLogger('agents')

# ---------------------------------------------------------------------------
# Column lists used when converting ORM objects to a DataFrame
# ---------------------------------------------------------------------------

SYMPTOM_COLS = [
    'optic_neuritis',
    'paresthesia',
    'weakness',
    'gait_instability',
    'vertigo',
    'fatigue',
    'bladder_issues',
    'cognitive_fog',
]

EXTENDED_COLS = [
    'vitamin_d_deficient',
    'infectious_mono_history',
    'smartform_neuro',
    'paths_like_function',
]

BASE_COLS = [
    'patient_id',
    'age',
    'sex',
    'mri_lesions',
    'note_has_ms_terms',
    'visit_count',
    'note_text',
    'lookalike_condition',
    'ms_diagnosis',
]

# Map from agent-expected column names to Patient model field names
_MODEL_FIELD_MAP = {
    'visit_count': 'visits_last_year',
    'note_text': 'note',
    'lookalike_condition': 'lookalike_dx',
    'ms_diagnosis': 'true_at_risk',
    'smartform_neuro': 'smartform_neuro_symptom_score',
    'paths_like_function': 'paths_like_function_score',
}

# Map coordinator autonomy_level int to RiskAssessment CharField values
_AUTONOMY_INT_TO_STR = {
    0: 'RECOMMEND_ONLY',
    1: 'RECOMMEND_ONLY',
    2: 'DRAFT_ORDER',
    3: 'AUTO_ORDER_WITH_GUARDRAILS',
}


def _patients_to_dataframe(patients_qs, limit: Optional[int] = None) -> pd.DataFrame:
    """Convert a Patient queryset into a pandas DataFrame."""
    cols = BASE_COLS + SYMPTOM_COLS + EXTENDED_COLS
    records = []
    qs = patients_qs.all()
    if limit:
        qs = qs[:limit]
    for p in qs:
        row = {}
        for col in cols:
            model_field = _MODEL_FIELD_MAP.get(col, col)
            row[col] = getattr(p, model_field, None)
        records.append(row)
    return pd.DataFrame(records)


def _patient_row_to_dict(row: pd.Series) -> Dict[str, Any]:
    """Convert a single DataFrame row to a plain dict for agent consumption."""
    d = row.to_dict()
    # Ensure native Python types for JSON serialization
    for k, v in d.items():
        if pd.isna(v):
            d[k] = None
        elif hasattr(v, 'item'):
            d[k] = v.item()
    return d


def _get_or_create_default_policy():
    """Get the active PolicyConfiguration or create a default one."""
    from patients.models import PolicyConfiguration
    policy_obj = PolicyConfiguration.objects.filter(is_active=True).first()
    if not policy_obj:
        policy_obj = PolicyConfiguration.objects.create(
            name='Default Policy',
            risk_review_threshold=settings.MS_RISK_POLICY.get('risk_review_threshold', 0.65),
            draft_order_threshold=settings.MS_RISK_POLICY.get('draft_order_threshold', 0.80),
            auto_order_threshold=settings.MS_RISK_POLICY.get('auto_order_threshold', 0.90),
            max_auto_actions_per_day=settings.MS_RISK_POLICY.get('max_auto_actions_per_day', 20),
            is_active=True,
            created_by='system',
        )
    return policy_obj


def build_patient_card(
    patient_data: Dict[str, Any],
    phenotyping_output: AgentOutput,
    notes_output: AgentOutput,
    safety_output: AgentOutput,
    coordinator_output: AgentOutput,
) -> Dict[str, Any]:
    """
    Assemble a single patient risk card from all agent outputs.

    This card is the primary data structure persisted as a
    RiskAssessment and displayed in the frontend.
    """
    risk_score = phenotyping_output.payload.get('risk_score', 0.0)
    feature_contributions = phenotyping_output.payload.get('feature_contributions', {})

    card = {
        'patient_id': patient_data.get('patient_id', 'UNKNOWN'),
        'age': patient_data.get('age'),
        'sex': patient_data.get('sex'),
        'risk_score': risk_score,
        'model_version': phenotyping_output.payload.get('model_version', 'v2'),
        'feature_contributions': feature_contributions,
        'top_features': sorted(
            [(k, v) for k, v in feature_contributions.items() if v > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:5],
        'lookalike_condition': patient_data.get('lookalike_condition', 'none'),
        'lookalike_penalty': feature_contributions.get('lookalike_penalty', 0.0),
        'note_ms_terms_flag': notes_output.payload.get('note_ms_terms_flag', False),
        'note_nonms_terms_flag': notes_output.payload.get('note_nonms_terms_flag', False),
        'note_excerpt': notes_output.payload.get('note_excerpt', ''),
        'ms_terms_found': notes_output.payload.get('ms_terms_found', []),
        'nonms_terms_found': notes_output.payload.get('nonms_terms_found', []),
        'safety_flags': safety_output.payload.get('flags', []),
        'safety_flag_count': safety_output.payload.get('flag_count', 0),
        'action': coordinator_output.payload.get('action', 'NO_ACTION'),
        'autonomy_level': coordinator_output.payload.get('autonomy_level', 0),
        'rationale': coordinator_output.payload.get('rationale', []),
        'ms_diagnosis_actual': patient_data.get('ms_diagnosis', False),
    }
    return card


def run_screening_workflow(
    policy_config_id: Optional[str] = None,
    patient_limit: Optional[int] = None,
) -> str:
    """
    Execute the full multi-agent MS screening pipeline.

    Steps
    -----
    1. Create a WorkflowRun record.
    2. Load patients from the DB into a DataFrame.
    3. Run RetrievalAgent to get candidate patient IDs.
    4. For each candidate, run the agent chain:
       PhenotypingAgentV2 -> NotesImagingAgent -> SafetyGovernanceAgent -> Coordinator
    5. Build patient cards and persist RiskAssessment records.
    6. Compute performance metrics (precision, recall, confusion counts).
    7. Update the WorkflowRun with results and metrics.

    Parameters
    ----------
    policy_config_id : str or None
        UUID string for a PolicyConfiguration record. If None,
        the active policy or system defaults are used.
    patient_limit : int or None
        Maximum number of patients to load (useful for dev/testing).

    Returns
    -------
    str
        The UUID (as string) of the created WorkflowRun.
    """
    from patients.models import Patient, WorkflowRun, RiskAssessment, PolicyConfiguration
    from .models import AgentExecution

    workflow_start = time.time()

    # ---- Load policy ----
    policy_obj = None
    if policy_config_id:
        try:
            policy_obj = PolicyConfiguration.objects.get(pk=policy_config_id)
        except PolicyConfiguration.DoesNotExist:
            logger.warning(f"PolicyConfiguration {policy_config_id} not found, using default.")

    if not policy_obj:
        policy_obj = _get_or_create_default_policy()

    policy = {
        'risk_review_threshold': policy_obj.risk_review_threshold,
        'draft_order_threshold': policy_obj.draft_order_threshold,
        'auto_order_threshold': policy_obj.auto_order_threshold,
        'max_auto_actions_per_day': policy_obj.max_auto_actions_per_day,
    }

    # ---- Create WorkflowRun ----
    wf_run = WorkflowRun.objects.create(
        policy=policy_obj,
        status='RUNNING',
    )
    logger.info(f"WorkflowRun {wf_run.id} started.")

    try:
        # ---- Step 2: Load patients ----
        patients_df = _patients_to_dataframe(Patient.objects, limit=patient_limit)
        if patients_df.empty:
            wf_run.status = 'COMPLETED'
            wf_run.error_message = 'No patients found'
            wf_run.save()
            return str(wf_run.id)

        wf_run.total_patients = len(patients_df)
        wf_run.save(update_fields=['total_patients'])

        # ---- Step 3: Retrieval ----
        retrieval = RetrievalAgent()
        retrieval_out = retrieval.timed_execute(patients_df)

        AgentExecution.objects.create(
            run=wf_run,
            agent_name='retrieval',
            patient_id_ref='ALL',
            payload=retrieval_out.payload,
            duration_ms=retrieval_out.duration_ms,
        )

        candidate_ids = retrieval_out.payload.get('candidate_ids', [])
        wf_run.candidates_found = len(candidate_ids)
        wf_run.save(update_fields=['candidates_found'])

        if not candidate_ids:
            wf_run.status = 'COMPLETED'
            wf_run.save(update_fields=['status'])
            return str(wf_run.id)

        # ---- Step 4: Per-candidate agent chain ----
        phenotyper = PhenotypingAgentV2()
        notes_agent = NotesImagingAgent()
        safety_agent = SafetyGovernanceAgent()
        coord = Coordinator(policy=policy)
        coord._reset_rate_limit()

        patient_cards: List[Dict[str, Any]] = []
        candidates_df = patients_df[patients_df['patient_id'].isin(candidate_ids)]

        auto_count = 0
        draft_count = 0
        recommend_count = 0
        total_flags = 0

        for _, row in candidates_df.iterrows():
            patient_data = _patient_row_to_dict(row)
            pid = patient_data.get('patient_id', 'UNKNOWN')

            # -- Phenotyping --
            pheno_out = phenotyper.timed_execute(patient_data)
            AgentExecution.objects.create(
                run=wf_run,
                agent_name='phenotyping',
                patient_id_ref=pid,
                payload=pheno_out.payload,
                duration_ms=pheno_out.duration_ms,
            )
            risk_score = pheno_out.payload.get('risk_score', 0.0)

            # -- Notes & Imaging --
            notes_out = notes_agent.timed_execute(patient_data)
            AgentExecution.objects.create(
                run=wf_run,
                agent_name='notes_imaging',
                patient_id_ref=pid,
                payload=notes_out.payload,
                duration_ms=notes_out.duration_ms,
            )

            # -- Safety --
            safety_out = safety_agent.timed_execute(patient_data, risk_score)
            AgentExecution.objects.create(
                run=wf_run,
                agent_name='safety_governance',
                patient_id_ref=pid,
                payload=safety_out.payload,
                duration_ms=safety_out.duration_ms,
            )

            # -- Coordinator --
            coord_out = coord.timed_execute(risk_score, safety_out.payload)
            AgentExecution.objects.create(
                run=wf_run,
                agent_name='coordinator',
                patient_id_ref=pid,
                payload=coord_out.payload,
                duration_ms=coord_out.duration_ms,
            )

            # -- Build card --
            card = build_patient_card(
                patient_data, pheno_out, notes_out, safety_out, coord_out,
            )
            patient_cards.append(card)

            # -- Track action counts --
            action = coord_out.payload.get('action', 'NO_ACTION')
            if action == 'AUTO_ORDER_MRI_AND_NOTIFY_NEURO':
                auto_count += 1
            elif action == 'DRAFT_MRI_ORDER':
                draft_count += 1
            elif action == 'RECOMMEND_NEURO_REVIEW':
                recommend_count += 1

            flag_count = safety_out.payload.get('flag_count', 0)
            total_flags += flag_count

            # -- Persist RiskAssessment --
            try:
                patient_obj = Patient.objects.get(patient_id=pid)
                autonomy_int = coord_out.payload.get('autonomy_level', 0)
                autonomy_str = _AUTONOMY_INT_TO_STR.get(autonomy_int, 'RECOMMEND_ONLY')

                RiskAssessment.objects.create(
                    patient=patient_obj,
                    run_id=wf_run.id,
                    risk_score=risk_score,
                    action=action,
                    autonomy_level=autonomy_str,
                    feature_contributions=pheno_out.payload.get('feature_contributions', {}),
                    flags=safety_out.payload.get('flags', []),
                    flag_count=flag_count,
                    rationale=coord_out.payload,
                    notes_analysis=notes_out.payload,
                    patient_card=card,
                )
            except Exception as e:
                logger.error(f"Failed to create RiskAssessment for {pid}: {e}")

        # ---- Step 6: Compute metrics ----
        metrics = _compute_metrics(patient_cards, policy)

        # ---- Step 7: Update WorkflowRun ----
        workflow_duration = time.time() - workflow_start
        flagged_count = auto_count + draft_count + recommend_count

        wf_run.status = 'COMPLETED'
        wf_run.flagged_count = flagged_count
        wf_run.precision = metrics.get('precision')
        wf_run.recall = metrics.get('recall')
        wf_run.auto_actions = auto_count
        wf_run.draft_actions = draft_count
        wf_run.recommend_actions = recommend_count
        wf_run.safety_flag_rate = (
            total_flags / len(patient_cards) if patient_cards else None
        )
        wf_run.duration_seconds = round(workflow_duration, 2)
        wf_run.save()

        logger.info(
            f"WorkflowRun {wf_run.id} completed in {workflow_duration:.1f}s. "
            f"Cards: {len(patient_cards)}  Metrics: {metrics}"
        )

        # ---- Generate notifications ----
        _create_workflow_notifications(
            wf_run, patient_cards, metrics, auto_count, draft_count,
            recommend_count, total_flags, workflow_duration,
        )

        return str(wf_run.id)

    except Exception as e:
        logger.exception(f"WorkflowRun {wf_run.id} failed: {e}")
        wf_run.status = 'FAILED'
        wf_run.error_message = str(e)
        wf_run.save()

        from core.models import Notification
        Notification.objects.create(
            title='Screening Workflow Failed',
            message=f'Workflow {str(wf_run.id)[:8]} failed: {str(e)[:200]}',
            severity='critical',
            category='workflow',
            metadata={'run_id': str(wf_run.id), 'error': str(e)[:500]},
        )
        raise


def _create_workflow_notifications(
    wf_run,
    patient_cards: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    auto_count: int,
    draft_count: int,
    recommend_count: int,
    total_flags: int,
    duration: float,
) -> None:
    """Create notifications summarising the completed workflow run."""
    from core.models import Notification

    run_id_short = str(wf_run.id)[:8]
    flagged = auto_count + draft_count + recommend_count
    total = len(patient_cards)

    # 1. Workflow completion summary
    Notification.objects.create(
        title='Screening Workflow Completed',
        message=(
            f'Workflow {run_id_short} screened {total} candidates in '
            f'{duration:.1f}s. Flagged: {flagged}, '
            f'Precision: {metrics.get("precision", 0):.2%}, '
            f'Recall: {metrics.get("recall", 0):.2%}.'
        ),
        severity='success',
        category='workflow',
        metadata={
            'run_id': str(wf_run.id),
            'total': total,
            'flagged': flagged,
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
        },
    )

    # 2. Auto-order actions taken
    if auto_count > 0:
        auto_patients = [
            c['patient_id'] for c in patient_cards
            if c.get('action') == 'AUTO_ORDER_MRI_AND_NOTIFY_NEURO'
        ]
        Notification.objects.create(
            title=f'{auto_count} Auto-Order MRI Action(s) Taken',
            message=(
                f'{auto_count} patient(s) had MRI orders automatically placed '
                f'and neurologist notified: {", ".join(auto_patients[:10])}'
                f'{"..." if len(auto_patients) > 10 else ""}.'
            ),
            severity='warning',
            category='auto_action',
            related_patient_id=auto_patients[0] if len(auto_patients) == 1 else '',
            metadata={
                'run_id': str(wf_run.id),
                'patient_ids': auto_patients,
            },
        )

    # 3. High-risk patients requiring review
    high_risk = [
        c for c in patient_cards
        if c.get('risk_score', 0) >= 0.80
    ]
    if high_risk:
        hr_ids = [c['patient_id'] for c in high_risk[:10]]
        Notification.objects.create(
            title=f'{len(high_risk)} High-Risk Patient(s) Identified',
            message=(
                f'{len(high_risk)} patient(s) scored >= 0.80 risk: '
                f'{", ".join(hr_ids)}{"..." if len(high_risk) > 10 else ""}. '
                f'Please review pending assessments.'
            ),
            severity='critical',
            category='high_risk',
            related_patient_id=hr_ids[0] if len(hr_ids) == 1 else '',
            metadata={
                'run_id': str(wf_run.id),
                'patient_ids': [c['patient_id'] for c in high_risk],
                'scores': {c['patient_id']: round(c['risk_score'], 3) for c in high_risk},
            },
        )

    # 4. Safety flags summary
    if total_flags > 0:
        flagged_patients = [
            c['patient_id'] for c in patient_cards
            if c.get('safety_flag_count', 0) > 0
        ]
        Notification.objects.create(
            title=f'Safety Flags on {len(flagged_patients)} Patient(s)',
            message=(
                f'{total_flags} safety flag(s) raised across '
                f'{len(flagged_patients)} patient(s). '
                f'Actions downgraded to RECOMMEND_ONLY where applicable.'
            ),
            severity='warning',
            category='safety',
            metadata={
                'run_id': str(wf_run.id),
                'total_flags': total_flags,
                'patient_ids': flagged_patients,
            },
        )

    # 5. Draft orders pending clinician sign-off
    if draft_count > 0:
        draft_patients = [
            c['patient_id'] for c in patient_cards
            if c.get('action') == 'DRAFT_MRI_ORDER'
        ]
        Notification.objects.create(
            title=f'{draft_count} Draft MRI Order(s) Pending Approval',
            message=(
                f'{draft_count} MRI order draft(s) require clinician sign-off: '
                f'{", ".join(draft_patients[:10])}'
                f'{"..." if len(draft_patients) > 10 else ""}.'
            ),
            severity='info',
            category='draft_order',
            metadata={
                'run_id': str(wf_run.id),
                'patient_ids': draft_patients,
            },
        )


def _compute_metrics(
    patient_cards: List[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute precision, recall, and confusion matrix counts from
    the set of patient cards produced by the workflow.

    A patient is considered "flagged" if its action is anything other
    than NO_ACTION.  Ground truth comes from ``ms_diagnosis_actual``.
    """
    review_thresh = policy.get('risk_review_threshold', 0.65)

    tp = 0  # flagged AND has MS
    fp = 0  # flagged AND no MS
    fn = 0  # not flagged AND has MS
    tn = 0  # not flagged AND no MS

    for card in patient_cards:
        flagged = card.get('action', 'NO_ACTION') != 'NO_ACTION'
        actual_ms = bool(card.get('ms_diagnosis_actual', False))

        if flagged and actual_ms:
            tp += 1
        elif flagged and not actual_ms:
            fp += 1
        elif not flagged and actual_ms:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'total_evaluated': len(patient_cards),
        'review_threshold_used': review_thresh,
    }

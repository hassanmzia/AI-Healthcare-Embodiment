"""REST API views for the MS Risk Lab application."""
import logging
from django.contrib.auth import authenticate
from django.utils import timezone
from django.db.models import Avg, Count, Q
from rest_framework import viewsets, status, filters
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, action, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend

from patients.models import Patient, RiskAssessment, PolicyConfiguration, WorkflowRun
from governance.models import GovernanceRule, ComplianceReport
from core.models import AuditLog, Notification
from .serializers import (
    PatientListSerializer, PatientDetailSerializer, RiskAssessmentSerializer,
    PolicyConfigurationSerializer, WorkflowRunSerializer,
    GovernanceRuleSerializer, ComplianceReportSerializer,
    AuditLogSerializer, NotificationSerializer,
    WorkflowTriggerSerializer, WhatIfSerializer, ReviewSerializer,
    LoginSerializer, RegisterSerializer, UserSerializer, ChangePasswordSerializer,
)
from analytics.services import (
    compute_workflow_metrics, subgroup_analysis, risk_distribution,
    action_distribution, autonomy_distribution, calibration_data,
    what_if_analysis,
)

logger = logging.getLogger(__name__)


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    return Response({'status': 'healthy', 'service': 'ms-risk-lab-api'})


# ─── Authentication Views ────────────────────────────────────────────────────

@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    serializer = LoginSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    user = authenticate(
        username=serializer.validated_data['username'],
        password=serializer.validated_data['password'],
    )
    if not user:
        return Response({'error': 'Invalid username or password.'}, status=status.HTTP_401_UNAUTHORIZED)
    if not user.is_active:
        return Response({'error': 'Account is disabled.'}, status=status.HTTP_403_FORBIDDEN)
    token, _ = Token.objects.get_or_create(user=user)
    return Response({
        'token': token.key,
        'user': UserSerializer(user).data,
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def register_view(request):
    serializer = RegisterSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    user = serializer.save()
    token = Token.objects.create(user=user)
    return Response({
        'token': token.key,
        'user': UserSerializer(user).data,
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])
def logout_view(request):
    request.user.auth_token.delete()
    return Response({'status': 'logged out'})


@api_view(['GET', 'PATCH'])
def profile_view(request):
    if request.method == 'GET':
        return Response(UserSerializer(request.user).data)
    serializer = UserSerializer(request.user, data=request.data, partial=True)
    serializer.is_valid(raise_exception=True)
    serializer.save()
    return Response(serializer.data)


@api_view(['POST'])
def change_password_view(request):
    serializer = ChangePasswordSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    if not request.user.check_password(serializer.validated_data['old_password']):
        return Response({'old_password': 'Current password is incorrect.'}, status=status.HTTP_400_BAD_REQUEST)
    request.user.set_password(serializer.validated_data['new_password'])
    request.user.save()
    # Rotate token after password change
    request.user.auth_token.delete()
    new_token = Token.objects.create(user=request.user)
    return Response({'status': 'password changed', 'token': new_token.key})


class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all()
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['sex', 'lookalike_dx', 'true_at_risk', 'has_mri', 'mri_lesions']
    search_fields = ['patient_id', 'note']
    ordering_fields = ['patient_id', 'age', 'visits_last_year', 'created_at']
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return PatientDetailSerializer
        return PatientListSerializer
    
    @action(detail=True, methods=['get'])
    def risk_history(self, request, pk=None):
        patient = self.get_object()
        assessments = patient.risk_assessments.order_by('-created_at')
        serializer = RiskAssessmentSerializer(assessments, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        total = Patient.objects.count()
        at_risk = Patient.objects.filter(true_at_risk=True).count()
        by_sex = list(Patient.objects.values('sex').annotate(count=Count('id')))
        by_dx = list(Patient.objects.values('lookalike_dx').annotate(count=Count('id')))
        avg_age = Patient.objects.aggregate(avg=Avg('age'))['avg']
        return Response({
            'total_patients': total,
            'at_risk_count': at_risk,
            'at_risk_rate': round(at_risk / total, 4) if total > 0 else 0,
            'by_sex': by_sex,
            'by_diagnosis': by_dx,
            'avg_age': round(avg_age, 1) if avg_age else 0,
        })


class RiskAssessmentViewSet(viewsets.ModelViewSet):
    queryset = RiskAssessment.objects.select_related('patient').all()
    serializer_class = RiskAssessmentSerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['action', 'autonomy_level', 'run_id']
    ordering_fields = ['risk_score', 'created_at', 'flag_count']
    
    @action(detail=True, methods=['post'])
    def review(self, request, pk=None):
        assessment = self.get_object()
        serializer = ReviewSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        assessment.reviewed_by = serializer.validated_data['reviewed_by']
        assessment.review_notes = serializer.validated_data.get('review_notes', '')
        assessment.reviewed_at = timezone.now()
        
        if 'override_action' in serializer.validated_data:
            old_action = assessment.action
            assessment.action = serializer.validated_data['override_action']
            AuditLog.objects.create(
                action_type='OVERRIDE',
                actor=assessment.reviewed_by,
                target_type='RiskAssessment',
                target_id=str(assessment.id),
                details={
                    'old_action': old_action,
                    'new_action': assessment.action,
                    'notes': assessment.review_notes,
                }
            )
        
        assessment.save()
        AuditLog.objects.create(
            action_type='MANUAL_REVIEW',
            actor=assessment.reviewed_by,
            target_type='RiskAssessment',
            target_id=str(assessment.id),
            details={'review_notes': assessment.review_notes}
        )
        return Response(RiskAssessmentSerializer(assessment).data)
    
    @action(detail=False, methods=['get'])
    def high_risk(self, request):
        threshold = float(request.query_params.get('threshold', 0.65))
        run_id = request.query_params.get('run_id')
        qs = self.queryset.filter(risk_score__gte=threshold)
        if run_id:
            qs = qs.filter(run_id=run_id)
        qs = qs.order_by('-risk_score')[:100]
        return Response(RiskAssessmentSerializer(qs, many=True).data)
    
    @action(detail=False, methods=['get'])
    def pending_review(self, request):
        qs = self.queryset.filter(
            reviewed_at__isnull=True,
            action__in=['RECOMMEND_NEURO_REVIEW', 'DRAFT_MRI_ORDER']
        ).order_by('-risk_score')[:100]
        return Response(RiskAssessmentSerializer(qs, many=True).data)


class PolicyConfigurationViewSet(viewsets.ModelViewSet):
    queryset = PolicyConfiguration.objects.all()
    serializer_class = PolicyConfigurationSerializer
    
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        policy = self.get_object()
        PolicyConfiguration.objects.filter(is_active=True).update(is_active=False)
        policy.is_active = True
        policy.save()
        AuditLog.objects.create(
            action_type='POLICY_CHANGE',
            actor=request.query_params.get('user', 'system'),
            target_type='PolicyConfiguration',
            target_id=str(policy.id),
            details={'name': policy.name}
        )
        return Response(PolicyConfigurationSerializer(policy).data)


class WorkflowRunViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = WorkflowRun.objects.select_related('policy').all()
    serializer_class = WorkflowRunSerializer
    
    @action(detail=False, methods=['post'])
    def trigger(self, request):
        serializer = WorkflowTriggerSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        requested_policy_id = serializer.validated_data.get('policy_id')
        policy_config_id = str(requested_policy_id) if requested_policy_id else None
        patient_limit = serializer.validated_data.get('patient_limit')

        # Resolve the actual policy that will be used
        if requested_policy_id:
            resolved_policy_id = str(requested_policy_id)
            resolved_policy_name = str(requested_policy_id)
        else:
            active_policy = PolicyConfiguration.objects.filter(is_active=True).first()
            if active_policy:
                resolved_policy_id = str(active_policy.id)
                resolved_policy_name = active_policy.name
            else:
                resolved_policy_id = 'default (will be created)'
                resolved_policy_name = 'Default Policy'

        # Resolve total patient count for audit when no limit specified
        if patient_limit is None:
            from patients.models import Patient
            total_patients = Patient.objects.count()
        else:
            total_patients = patient_limit

        from agents.tasks import run_screening_workflow_task
        task = run_screening_workflow_task.delay(
            policy_config_id=policy_config_id,
            patient_limit=patient_limit,
        )

        AuditLog.objects.create(
            action_type='AGENT_RUN',
            actor='api',
            target_type='WorkflowRun',
            target_id=task.id,
            details={
                'task_id': task.id,
                'policy_id': resolved_policy_id,
                'policy_name': resolved_policy_name,
                'patient_limit': patient_limit if patient_limit else f'all ({total_patients})',
            }
        )
        return Response({'task_id': task.id, 'status': 'queued'}, status=status.HTTP_202_ACCEPTED)
    
    @action(detail=True, methods=['get'])
    def metrics(self, request, pk=None):
        run = self.get_object()
        metrics = compute_workflow_metrics(run.id)
        return Response(metrics)
    
    @action(detail=True, methods=['get'])
    def risk_distribution(self, request, pk=None):
        run = self.get_object()
        return Response(risk_distribution(run.id))
    
    @action(detail=True, methods=['get'])
    def action_distribution(self, request, pk=None):
        run = self.get_object()
        return Response(action_distribution(run.id))
    
    @action(detail=True, methods=['get'])
    def autonomy_distribution(self, request, pk=None):
        run = self.get_object()
        return Response(autonomy_distribution(run.id))
    
    @action(detail=True, methods=['get'])
    def calibration(self, request, pk=None):
        run = self.get_object()
        return Response(calibration_data(run.id))
    
    @action(detail=True, methods=['get'])
    def fairness(self, request, pk=None):
        run = self.get_object()
        group_by = request.query_params.get('group_by', 'sex')
        return Response(subgroup_analysis(run.id, group_by))


class WhatIfView(APIView):
    def post(self, request):
        serializer = WhatIfSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        run_id = serializer.validated_data.pop('run_id')
        overrides = {k: v for k, v in serializer.validated_data.items() if v is not None}
        
        result = what_if_analysis(run_id, overrides)
        return Response(result)


class GovernanceRuleViewSet(viewsets.ModelViewSet):
    queryset = GovernanceRule.objects.all()
    serializer_class = GovernanceRuleSerializer


class ComplianceReportViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ComplianceReport.objects.all()
    serializer_class = ComplianceReportSerializer
    
    @action(detail=False, methods=['post'])
    def generate(self, request):
        run_id = request.data.get('run_id')
        if not run_id:
            return Response({'error': 'run_id required'}, status=400)
        from analytics.tasks import generate_compliance_report
        task = generate_compliance_report.delay(run_id)
        return Response({'task_id': task.id, 'status': 'generating'})


class AuditLogViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AuditLog.objects.all()
    serializer_class = AuditLogSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['action_type', 'actor']


class NotificationViewSet(viewsets.ModelViewSet):
    queryset = Notification.objects.all()
    serializer_class = NotificationSerializer
    
    @action(detail=True, methods=['post'])
    def mark_read(self, request, pk=None):
        notification = self.get_object()
        notification.is_read = True
        notification.save()
        return Response(NotificationSerializer(notification).data)
    
    @action(detail=False, methods=['post'])
    def mark_all_read(self, request):
        Notification.objects.filter(is_read=False).update(is_read=True)
        return Response({'status': 'all marked read'})
    
    @action(detail=False, methods=['get'])
    def unread_count(self, request):
        count = Notification.objects.filter(is_read=False).count()
        return Response({'unread_count': count})


class DashboardView(APIView):
    """Aggregated dashboard data endpoint."""
    
    def get(self, request):
        total_patients = Patient.objects.count()
        at_risk = Patient.objects.filter(true_at_risk=True).count()
        
        latest_run = WorkflowRun.objects.filter(status='COMPLETED').first()
        
        run_data = None
        if latest_run:
            assessments = RiskAssessment.objects.filter(run_id=latest_run.id)
            run_data = {
                'run_id': str(latest_run.id),
                'status': latest_run.status,
                'total_assessed': assessments.count(),
                'flagged': assessments.exclude(action='NO_ACTION').count(),
                'precision': latest_run.precision,
                'recall': latest_run.recall,
                'auto_actions': latest_run.auto_actions,
                'draft_actions': latest_run.draft_actions,
                'recommend_actions': latest_run.recommend_actions,
                'created_at': latest_run.created_at,
            }
        
        pending_reviews = RiskAssessment.objects.filter(
            reviewed_at__isnull=True,
            action__in=['RECOMMEND_NEURO_REVIEW', 'DRAFT_MRI_ORDER']
        ).count()
        
        unread_notifications = Notification.objects.filter(is_read=False).count()
        
        recent_runs = WorkflowRunSerializer(
            WorkflowRun.objects.all()[:5], many=True
        ).data
        
        return Response({
            'total_patients': total_patients,
            'at_risk_count': at_risk,
            'at_risk_rate': round(at_risk / total_patients, 4) if total_patients > 0 else 0,
            'latest_run': run_data,
            'pending_reviews': pending_reviews,
            'unread_notifications': unread_notifications,
            'recent_runs': recent_runs,
            'total_workflow_runs': WorkflowRun.objects.count(),
        })

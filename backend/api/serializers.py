"""REST API serializers."""
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers
from patients.models import Patient, RiskAssessment, PolicyConfiguration, WorkflowRun
from governance.models import GovernanceRule, ComplianceReport
from core.models import AuditLog, Notification


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password_confirm', 'first_name', 'last_name']

    def validate(self, attrs):
        if attrs['password'] != attrs.pop('password_confirm'):
            raise serializers.ValidationError({'password_confirm': 'Passwords do not match.'})
        if User.objects.filter(username=attrs['username']).exists():
            raise serializers.ValidationError({'username': 'Username already taken.'})
        if attrs.get('email') and User.objects.filter(email=attrs['email']).exists():
            raise serializers.ValidationError({'email': 'Email already registered.'})
        return attrs

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'date_joined', 'is_staff']
        read_only_fields = ['id', 'username', 'date_joined', 'is_staff']


class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(write_only=True)
    new_password = serializers.CharField(write_only=True, validators=[validate_password])
    new_password_confirm = serializers.CharField(write_only=True)

    def validate(self, attrs):
        if attrs['new_password'] != attrs['new_password_confirm']:
            raise serializers.ValidationError({'new_password_confirm': 'Passwords do not match.'})
        return attrs


class PatientListSerializer(serializers.ModelSerializer):
    symptom_count = serializers.ReadOnlyField()
    
    class Meta:
        model = Patient
        fields = [
            'id', 'patient_id', 'age', 'sex', 'visits_last_year',
            'lookalike_dx', 'has_mri', 'mri_lesions', 'note_has_ms_terms',
            'true_at_risk', 'symptom_count', 'optic_neuritis', 'paresthesia',
            'weakness', 'gait_instability', 'vertigo', 'fatigue',
            'bladder_issues', 'cognitive_fog', 'vitamin_d_ngml',
            'vitamin_d_deficient', 'infectious_mono_history',
            'smartform_neuro_symptom_score', 'paths_like_function_score',
            'created_at',
        ]


class PatientDetailSerializer(serializers.ModelSerializer):
    symptom_count = serializers.ReadOnlyField()
    risk_assessments = serializers.SerializerMethodField()
    
    class Meta:
        model = Patient
        fields = '__all__'
    
    def get_risk_assessments(self, obj):
        assessments = obj.risk_assessments.order_by('-created_at')[:10]
        return RiskAssessmentSerializer(assessments, many=True).data


class RiskAssessmentSerializer(serializers.ModelSerializer):
    patient_display = serializers.CharField(source='patient.patient_id', read_only=True)
    
    class Meta:
        model = RiskAssessment
        fields = [
            'id', 'patient', 'patient_display', 'run_id', 'risk_score',
            'action', 'autonomy_level', 'feature_contributions', 'flags',
            'flag_count', 'rationale', 'notes_analysis', 'llm_summary',
            'patient_card', 'reviewed_by', 'review_notes', 'reviewed_at',
            'created_at',
        ]


class PolicyConfigurationSerializer(serializers.ModelSerializer):
    class Meta:
        model = PolicyConfiguration
        fields = '__all__'


class WorkflowRunSerializer(serializers.ModelSerializer):
    policy_name = serializers.CharField(source='policy.name', read_only=True)
    
    class Meta:
        model = WorkflowRun
        fields = [
            'id', 'policy', 'policy_name', 'status', 'total_patients',
            'candidates_found', 'flagged_count', 'precision', 'recall',
            'auto_actions', 'draft_actions', 'recommend_actions',
            'safety_flag_rate', 'duration_seconds', 'error_message',
            'created_at', 'updated_at',
        ]


class GovernanceRuleSerializer(serializers.ModelSerializer):
    class Meta:
        model = GovernanceRule
        fields = '__all__'


class ComplianceReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = ComplianceReport
        fields = '__all__'


class AuditLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = AuditLog
        fields = '__all__'


class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = '__all__'


class WorkflowTriggerSerializer(serializers.Serializer):
    policy_id = serializers.UUIDField(required=False)
    patient_limit = serializers.IntegerField(required=False, min_value=1, max_value=10000)


class WhatIfSerializer(serializers.Serializer):
    run_id = serializers.UUIDField()
    risk_review_threshold = serializers.FloatField(required=False, min_value=0, max_value=1)
    draft_order_threshold = serializers.FloatField(required=False, min_value=0, max_value=1)
    auto_order_threshold = serializers.FloatField(required=False, min_value=0, max_value=1)
    max_auto_actions_per_day = serializers.IntegerField(required=False, min_value=0)


class ReviewSerializer(serializers.Serializer):
    reviewed_by = serializers.CharField(max_length=200)
    review_notes = serializers.CharField(allow_blank=True)
    override_action = serializers.ChoiceField(
        choices=['NO_ACTION', 'RECOMMEND_NEURO_REVIEW', 'DRAFT_MRI_ORDER', 'AUTO_ORDER_MRI_AND_NOTIFY_NEURO'],
        required=False
    )

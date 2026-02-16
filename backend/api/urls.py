"""API URL configuration."""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'patients', views.PatientViewSet)
router.register(r'assessments', views.RiskAssessmentViewSet)
router.register(r'policies', views.PolicyConfigurationViewSet)
router.register(r'workflows', views.WorkflowRunViewSet)
router.register(r'governance-rules', views.GovernanceRuleViewSet)
router.register(r'compliance-reports', views.ComplianceReportViewSet)
router.register(r'audit-logs', views.AuditLogViewSet)
router.register(r'notifications', views.NotificationViewSet)

urlpatterns = [
    path('health/', views.health_check),
    path('auth/login/', views.login_view),
    path('auth/register/', views.register_view),
    path('auth/logout/', views.logout_view),
    path('auth/profile/', views.profile_view),
    path('auth/change-password/', views.change_password_view),
    path('dashboard/', views.DashboardView.as_view()),
    path('what-if/', views.WhatIfView.as_view()),
    path('', include(router.urls)),
]

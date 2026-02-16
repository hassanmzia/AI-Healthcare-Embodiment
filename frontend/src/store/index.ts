import { create } from 'zustand';
import type { DashboardData, WorkflowRun, Notification, User } from '../types';

interface AppState {
  // Auth
  isAuthenticated: boolean;
  user: User | null;
  setAuth: (user: User, token: string) => void;
  clearAuth: () => void;

  // Dashboard
  dashboard: DashboardData | null;
  dashboardLoading: boolean;
  setDashboard: (data: DashboardData) => void;
  setDashboardLoading: (loading: boolean) => void;

  // Active workflow run
  activeRunId: string | null;
  setActiveRunId: (id: string | null) => void;

  // Notifications
  notifications: Notification[];
  unreadCount: number;
  setNotifications: (notifications: Notification[]) => void;
  setUnreadCount: (count: number) => void;

  // Theme
  darkMode: boolean;
  toggleDarkMode: () => void;

  // Sidebar
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
}

const storedUser = localStorage.getItem('auth_user');
const storedToken = localStorage.getItem('auth_token');

export const useAppStore = create<AppState>((set) => ({
  isAuthenticated: !!storedToken,
  user: storedUser ? JSON.parse(storedUser) : null,
  setAuth: (user, token) => {
    localStorage.setItem('auth_token', token);
    localStorage.setItem('auth_user', JSON.stringify(user));
    set({ isAuthenticated: true, user });
  },
  clearAuth: () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
    set({ isAuthenticated: false, user: null });
  },

  dashboard: null,
  dashboardLoading: false,
  setDashboard: (data) => set({ dashboard: data }),
  setDashboardLoading: (loading) => set({ dashboardLoading: loading }),

  activeRunId: null,
  setActiveRunId: (id) => set({ activeRunId: id }),

  notifications: [],
  unreadCount: 0,
  setNotifications: (notifications) => set({ notifications }),
  setUnreadCount: (count) => set({ unreadCount: count }),

  darkMode: false,
  toggleDarkMode: () => set((state) => ({ darkMode: !state.darkMode })),

  sidebarOpen: typeof window !== 'undefined' && window.innerWidth < 900 ? false : true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
}));

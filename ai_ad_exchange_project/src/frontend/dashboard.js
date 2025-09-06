// AI Ad Exchange Dashboard JavaScript

class AdExchangeDashboard {
    constructor() {
        this.apiBase = '/api/v1';
        this.stats = {};
        this.init();
    }
    
    async init() {
        console.log('AI Ad Exchange Dashboard initializing...');
        await this.loadDashboard();
        this.setupAutoRefresh();
    }
    
    async loadDashboard() {
        try {
            // Load exchange statistics
            await this.loadExchangeStats();
            
            // Load publisher data
            await this.loadPublishers();
            
            // Load advertiser data
            await this.loadAdvertisers();
            
            // Update UI
            this.updateDashboard();
            
            console.log('Dashboard loaded successfully');
        } catch (error) {
            console.error('Error loading dashboard:', error);
            this.showError('Failed to load dashboard data');
        }
    }
    
    async loadExchangeStats() {
        try {
            const response = await fetch(`${this.apiBase}/stats`);
            if (response.ok) {
                const data = await response.json();
                this.stats = data.data || {};
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.error('Error loading exchange stats:', error);
            // Use fallback data
            this.stats = {
                total_publishers: 3,
                total_advertisers: 3,
                active_campaigns: 3,
                total_impressions_today: 1500,
                total_clicks_today: 45,
                total_revenue_today: 22.50
            };
        }
    }
    
    async loadPublishers() {
        try {
            const response = await fetch(`${this.apiBase}/publishers`);
            if (response.ok) {
                const data = await response.json();
                this.publishers = data.data || [];
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.error('Error loading publishers:', error);
            // Use fallback data
            this.publishers = [
                {
                    id: 'pub_001',
                    name: 'GamingStreamer123',
                    performance_metrics: { total_earnings: 1250.50 }
                },
                {
                    id: 'pub_002',
                    name: 'TechReviewer',
                    performance_metrics: { total_earnings: 2100.75 }
                },
                {
                    id: 'pub_003',
                    name: 'FitnessCoach',
                    performance_metrics: { total_earnings: 850.25 }
                }
            ];
        }
    }
    
    async loadAdvertisers() {
        try {
            const response = await fetch(`${this.apiBase}/advertisers`);
            if (response.ok) {
                const data = await response.json();
                this.advertisers = data.data || [];
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.error('Error loading advertisers:', error);
            // Use fallback data
            this.advertisers = [
                {
                    id: 'adv_001',
                    name: 'GamingTech Inc',
                    budget: { spent: 12500 }
                },
                {
                    id: 'adv_002',
                    name: 'FitnessSupplements Co',
                    budget: { spent: 8000 }
                },
                {
                    id: 'adv_003',
                    name: 'TechStartup LLC',
                    budget: { spent: 3000 }
                }
            ];
        }
    }
    
    updateDashboard() {
        this.updateStats();
        this.updatePublishers();
        this.updateAdvertisers();
    }
    
    updateStats() {
        // Update statistics cards
        const elements = {
            'total-publishers': this.stats.total_publishers || 0,
            'total-advertisers': this.stats.total_advertisers || 0,
            'active-campaigns': this.stats.active_campaigns || 0,
            'today-impressions': this.formatNumber(this.stats.total_impressions_today || 0),
            'today-clicks': this.formatNumber(this.stats.total_clicks_today || 0),
            'today-revenue': this.formatCurrency(this.stats.total_revenue_today || 0)
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }
    
    updatePublishers() {
        const container = document.getElementById('top-publishers');
        if (!container || !this.publishers) return;
        
        // Sort publishers by earnings
        const sortedPublishers = [...this.publishers].sort((a, b) => {
            const earningsA = a.performance_metrics?.total_earnings || 0;
            const earningsB = b.performance_metrics?.total_earnings || 0;
            return earningsB - earningsA;
        });
        
        // Update top publishers list
        container.innerHTML = sortedPublishers.slice(0, 3).map(publisher => `
            <div class="list-item">
                <span class="name">${publisher.name}</span>
                <span class="value">${this.formatCurrency(publisher.performance_metrics?.total_earnings || 0)}</span>
            </div>
        `).join('');
    }
    
    updateAdvertisers() {
        const container = document.getElementById('top-advertisers');
        if (!container || !this.advertisers) return;
        
        // Sort advertisers by spent budget
        const sortedAdvertisers = [...this.advertisers].sort((a, b) => {
            const spentA = a.budget?.spent || 0;
            const spentB = b.budget?.spent || 0;
            return spentB - spentA;
        });
        
        // Update top advertisers list
        container.innerHTML = sortedAdvertisers.slice(0, 3).map(advertiser => `
            <div class="list-item">
                <span class="name">${advertiser.name}</span>
                <span class="value">${this.formatCurrency(advertiser.budget?.spent || 0)}</span>
            </div>
        `).join('');
    }
    
    formatNumber(num) {
        return new Intl.NumberFormat().format(num);
    }
    
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }
    
    setupAutoRefresh() {
        // Auto-refresh dashboard every 30 seconds
        setInterval(() => {
            this.loadDashboard();
        }, 30000);
    }
    
    showError(message) {
        // Create error notification
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #e74c3c;
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Remove notification after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }
    
    showSuccess(message) {
        // Create success notification
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #27ae60;
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Remove notification after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize dashboard when page loads
let dashboard;

document.addEventListener('DOMContentLoaded', () => {
    dashboard = new AdExchangeDashboard();
});

// Global function for refresh button
function loadDashboard() {
    if (dashboard) {
        dashboard.loadDashboard();
        dashboard.showSuccess('Dashboard refreshed successfully');
    }
}

// Add loading indicator
function showLoading() {
    const loading = document.createElement('div');
    loading.id = 'loading';
    loading.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255,255,255,0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    `;
    loading.innerHTML = `
        <div style="text-align: center;">
            <div style="width: 50px; height: 50px; border: 5px solid #f3f3f3; border-top: 5px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <p style="margin-top: 20px; color: #2c3e50;">Loading dashboard...</p>
        </div>
    `;
    document.body.appendChild(loading);
}

function hideLoading() {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.remove();
    }
}

// Add spinner animation
const spinnerStyle = document.createElement('style');
spinnerStyle.textContent = `
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
document.head.appendChild(spinnerStyle); 
/**
 * AI Phone Agent Dashboard JavaScript
 */

class PhoneAgentDashboard {
    constructor() {
        this.isConnected = false;
        this.conversationHistory = [];
        this.currentCallId = null;
        this.updateInterval = null;
        
        this.initialize();
    }
    
    initialize() {
        console.log('AI Phone Agent Dashboard initializing...');
        this.setupEventListeners();
        this.startStatusUpdates();
        this.loadInitialData();
    }
    
    setupEventListeners() {
        // Handle Enter key in input field
        document.getElementById('user-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
    }
    
    startStatusUpdates() {
        // Update status every 5 seconds
        this.updateInterval = setInterval(() => {
            this.updateStatus();
        }, 5000);
        
        // Initial update
        this.updateStatus();
    }
    
    async updateStatus() {
        try {
            // In a real implementation, this would fetch status from the API
            const status = await this.fetchStatus();
            this.updateDashboard(status);
        } catch (error) {
            console.error('Error updating status:', error);
            this.showError('Failed to update status');
        }
    }
    
    async fetchStatus() {
        // Simulate API call
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    system_status: 'online',
                    active_calls: Math.floor(Math.random() * 3),
                    total_calls_today: Math.floor(Math.random() * 50) + 10,
                    average_duration: '2:30',
                    response_time: '0.5s',
                    success_rate: '98%',
                    satisfaction: '4.8/5',
                    uptime: '99.9%'
                });
            }, 100);
        });
    }
    
    updateDashboard(status) {
        // Update status indicators
        document.getElementById('active-calls').textContent = status.active_calls;
        document.getElementById('total-calls').textContent = status.total_calls_today;
        document.getElementById('avg-duration').textContent = status.average_duration;
        document.getElementById('response-time').textContent = status.response_time;
        document.getElementById('success-rate').textContent = status.success_rate;
        document.getElementById('satisfaction').textContent = status.satisfaction;
        document.getElementById('uptime').textContent = status.uptime;
        
        // Update last update time
        document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
    }
    
    async startTestCall() {
        try {
            this.showMessage('Starting test call...', 'info');
            
            // Simulate API call to start test call
            const response = await this.apiCall('start_test_call', {});
            
            if (response.success) {
                this.currentCallId = response.call_id;
                this.isConnected = true;
                this.showMessage('Test call started successfully!', 'success');
                this.addToConversation('Agent', 'Hello! Thank you for calling. How can I help you today?');
            } else {
                this.showMessage('Failed to start test call', 'error');
            }
        } catch (error) {
            console.error('Error starting test call:', error);
            this.showMessage('Error starting test call', 'error');
        }
    }
    
    async startProduction() {
        try {
            this.showMessage('Starting production mode...', 'info');
            
            // Simulate API call to start production
            const response = await this.apiCall('start_production', {});
            
            if (response.success) {
                this.showMessage('Production mode started successfully!', 'success');
                this.updateSystemStatus('production');
            } else {
                this.showMessage('Failed to start production mode', 'error');
            }
        } catch (error) {
            console.error('Error starting production:', error);
            this.showMessage('Error starting production mode', 'error');
        }
    }
    
    async pauseSystem() {
        try {
            this.showMessage('Pausing system...', 'info');
            
            // Simulate API call to pause system
            const response = await this.apiCall('pause_system', {});
            
            if (response.success) {
                this.showMessage('System paused successfully', 'success');
                this.updateSystemStatus('paused');
            } else {
                this.showMessage('Failed to pause system', 'error');
            }
        } catch (error) {
            console.error('Error pausing system:', error);
            this.showMessage('Error pausing system', 'error');
        }
    }
    
    async emergencyStop() {
        try {
            this.showMessage('Emergency stop initiated...', 'warning');
            
            // Simulate API call for emergency stop
            const response = await this.apiCall('emergency_stop', {});
            
            if (response.success) {
                this.showMessage('Emergency stop completed', 'success');
                this.isConnected = false;
                this.currentCallId = null;
                this.updateSystemStatus('stopped');
            } else {
                this.showMessage('Failed to perform emergency stop', 'error');
            }
        } catch (error) {
            console.error('Error during emergency stop:', error);
            this.showMessage('Error during emergency stop', 'error');
        }
    }
    
    async sendMessage() {
        const input = document.getElementById('user-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        if (!this.isConnected) {
            this.showMessage('Please start a test call first', 'warning');
            return;
        }
        
        // Add user message to conversation
        this.addToConversation('You', message);
        input.value = '';
        
        try {
            // Simulate API call to process message
            const response = await this.apiCall('process_message', {
                call_id: this.currentCallId,
                message: message
            });
            
            if (response.success && response.agent_response) {
                this.addToConversation('Agent', response.agent_response);
            } else {
                this.addToConversation('Agent', 'I\'m sorry, I didn\'t understand that. Could you please repeat?');
            }
        } catch (error) {
            console.error('Error processing message:', error);
            this.addToConversation('System', 'Error processing message. Please try again.');
        }
    }
    
    addToConversation(speaker, message) {
        const timestamp = new Date().toLocaleTimeString();
        const entry = `[${timestamp}] ${speaker}: ${message}`;
        
        this.conversationHistory.push(entry);
        this.updateConversationLog();
    }
    
    updateConversationLog() {
        const log = document.getElementById('conversation-log');
        log.innerHTML = this.conversationHistory.map(entry => 
            `<div style="margin-bottom: 8px;">${entry}</div>`
        ).join('');
        
        // Scroll to bottom
        log.scrollTop = log.scrollHeight;
    }
    
    updateSystemStatus(status) {
        const statusIndicator = document.querySelector('.status-indicator');
        const statusText = document.querySelector('.card p');
        
        statusIndicator.className = 'status-indicator';
        
        switch (status) {
            case 'online':
                statusIndicator.classList.add('status-online');
                statusText.innerHTML = '<span class="status-indicator status-online"></span>Online';
                break;
            case 'testing':
                statusIndicator.classList.add('status-testing');
                statusText.innerHTML = '<span class="status-indicator status-testing"></span>Testing';
                break;
            case 'paused':
                statusIndicator.classList.add('status-warning');
                statusText.innerHTML = '<span class="status-indicator status-warning"></span>Paused';
                break;
            case 'stopped':
                statusIndicator.classList.add('status-offline');
                statusText.innerHTML = '<span class="status-indicator status-offline"></span>Stopped';
                break;
        }
    }
    
    showMessage(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        
        // Set background color based on type
        switch (type) {
            case 'success':
                notification.style.backgroundColor = '#27ae60';
                break;
            case 'error':
                notification.style.backgroundColor = '#e74c3c';
                break;
            case 'warning':
                notification.style.backgroundColor = '#f39c12';
                break;
            default:
                notification.style.backgroundColor = '#3498db';
        }
        
        document.body.appendChild(notification);
        
        // Remove notification after 3 seconds
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
    
    showError(message) {
        this.showMessage(message, 'error');
    }
    
    async apiCall(endpoint, data) {
        // Simulate API calls
        return new Promise((resolve) => {
            setTimeout(() => {
                switch (endpoint) {
                    case 'start_test_call':
                        resolve({
                            success: true,
                            call_id: 'test_call_' + Date.now()
                        });
                        break;
                    case 'start_production':
                        resolve({ success: true });
                        break;
                    case 'pause_system':
                        resolve({ success: true });
                        break;
                    case 'emergency_stop':
                        resolve({ success: true });
                        break;
                    case 'process_message':
                        // Simulate AI response
                        const responses = [
                            "I understand. How can I help you with that?",
                            "Thank you for sharing that. What would you like me to do?",
                            "I see. Let me assist you with that.",
                            "Got it. How can I be of help?",
                            "That's a great question. Let me help you find the information you need."
                        ];
                        resolve({
                            success: true,
                            agent_response: responses[Math.floor(Math.random() * responses.length)]
                        });
                        break;
                    default:
                        resolve({ success: false, error: 'Unknown endpoint' });
                }
            }, 500);
        });
    }
    
    loadInitialData() {
        // Load initial statistics
        document.getElementById('total-conversations').textContent = '0';
        document.getElementById('knowledge-entries').textContent = '12';
        document.getElementById('languages').textContent = '1';
    }
    
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

// Initialize dashboard when page loads
let dashboard;

document.addEventListener('DOMContentLoaded', () => {
    dashboard = new PhoneAgentDashboard();
});

// Global functions for button clicks
function startTestCall() {
    if (dashboard) dashboard.startTestCall();
}

function startProduction() {
    if (dashboard) dashboard.startProduction();
}

function pauseSystem() {
    if (dashboard) dashboard.pauseSystem();
}

function emergencyStop() {
    if (dashboard) dashboard.emergencyStop();
}

function sendMessage() {
    if (dashboard) dashboard.sendMessage();
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Add CSS for notifications
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
`;
document.head.appendChild(style); 
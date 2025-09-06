"""
Focus Agent - Manages workflow and concentration
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Task:
    """Represents a task to be managed"""
    task_id: str
    task_type: str  # "token_scan", "ai_analysis", "trade_execution", "risk_check", "performance_review"
    priority: int  # 1-5, higher is more important
    description: str
    data: Dict[str, Any]
    created_at: datetime
    deadline: Optional[datetime] = None
    status: str = "pending"  # "pending", "in_progress", "completed", "failed", "cancelled"
    assigned_to: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowState:
    """Represents the current workflow state"""
    current_phase: str
    active_tasks: List[str]
    completed_tasks: List[str]
    failed_tasks: List[str]
    performance_metrics: Dict[str, Any]
    last_update: datetime


@dataclass
class SystemHealth:
    """Represents system health metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    api_response_time: float
    error_rate: float
    uptime: float
    timestamp: datetime


class FocusAgent:
    """
    Workflow management and concentration agent
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.max_concurrent_tasks = config.get("max_concurrent_tasks", 5)
        self.task_timeout = config.get("task_timeout", 300)  # 5 minutes
        self.health_check_interval = config.get("health_check_interval", 60)  # 1 minute
        self.performance_review_interval = config.get("performance_review_interval", 3600)  # 1 hour
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.active_tasks: Set[str] = set()
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        
        # Workflow state
        self.current_phase = "initialization"
        self.workflow_state = WorkflowState(
            current_phase=self.current_phase,
            active_tasks=[],
            completed_tasks=[],
            failed_tasks=[],
            performance_metrics={},
            last_update=datetime.now()
        )
        
        # System health
        self.system_health = SystemHealth(
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            network_latency=0.0,
            api_response_time=0.0,
            error_rate=0.0,
            uptime=0.0,
            timestamp=datetime.now()
        )
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.failed_tasks_count = 0
        self.avg_task_completion_time = 0.0
        
        # State management
        self.is_running = False
        self.health_monitor_task = None
        self.performance_review_task = None
        
    async def initialize(self):
        """Initialize the focus agent"""
        logger.info("Initializing Focus Agent...")
        
        try:
            # Start health monitoring
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            # Start performance review
            self.performance_review_task = asyncio.create_task(self._performance_review_loop())
            
            logger.info("Focus Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Focus Agent: {e}")
            return False
    
    async def start(self):
        """Start the focus agent"""
        if not self.enabled:
            logger.warning("Focus Agent is disabled")
            return
        
        if self.is_running:
            logger.warning("Focus Agent is already running")
            return
        
        logger.info("Starting Focus Agent...")
        self.is_running = True
        self.current_phase = "active"
        
        try:
            while self.is_running:
                await self._workflow_cycle()
                await asyncio.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in focus agent main loop: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the focus agent"""
        logger.info("Stopping Focus Agent...")
        self.is_running = False
        self.current_phase = "shutdown"
        
        # Cancel background tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        if self.performance_review_task:
            self.performance_review_task.cancel()
        
        logger.info("Focus Agent stopped")
    
    async def add_task(self, task_type: str, priority: int, description: str, 
                      data: Dict[str, Any], deadline: Optional[datetime] = None) -> str:
        """Add a new task to the queue"""
        task_id = f"{task_type}_{int(time.time())}_{len(self.tasks)}"
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            description=description,
            data=data,
            created_at=datetime.now(),
            deadline=deadline
        )
        
        self.tasks[task_id] = task
        logger.info(f"Added task: {task_id} - {description}")
        
        return task_id
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a specific task"""
        return self.tasks.get(task_id)
    
    async def update_task_status(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None):
        """Update task status"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            task.result = result
            
            if status == "completed":
                self.completed_tasks.append(task_id)
                self.successful_tasks += 1
                if task_id in self.active_tasks:
                    self.active_tasks.remove(task_id)
                    
            elif status == "failed":
                self.failed_tasks.append(task_id)
                self.failed_tasks_count += 1
                if task_id in self.active_tasks:
                    self.active_tasks.remove(task_id)
            
            logger.info(f"Updated task {task_id} status to {status}")
    
    async def _workflow_cycle(self):
        """Main workflow cycle"""
        try:
            # 1. Check system health
            await self._check_system_health()
            
            # 2. Process pending tasks
            await self._process_pending_tasks()
            
            # 3. Monitor active tasks
            await self._monitor_active_tasks()
            
            # 4. Update workflow state
            await self._update_workflow_state()
            
        except Exception as e:
            logger.error(f"Error in workflow cycle: {e}")
    
    async def _check_system_health(self):
        """Check system health metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network latency (simple ping test)
            network_latency = await self._measure_network_latency()
            
            # API response time (average of recent calls)
            api_response_time = self._get_avg_api_response_time()
            
            # Error rate
            error_rate = self._calculate_error_rate()
            
            # Uptime
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            self.system_health = SystemHealth(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_latency=network_latency,
                api_response_time=api_response_time,
                error_rate=error_rate,
                uptime=uptime,
                timestamp=datetime.now()
            )
            
            # Check for health issues
            await self._check_health_issues()
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    async def _measure_network_latency(self) -> float:
        """Measure network latency"""
        try:
            import aiohttp
            import time
            
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.mainnet-beta.solana.com', timeout=5) as response:
                    if response.status == 200:
                        return (time.time() - start_time) * 1000  # Convert to milliseconds
            return 1000.0  # Default high latency if failed
        except:
            return 1000.0
    
    def _get_avg_api_response_time(self) -> float:
        """Get average API response time"""
        # This would typically track recent API calls
        # For now, return a default value
        return 500.0  # milliseconds
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate"""
        total_tasks = self.successful_tasks + self.failed_tasks_count
        if total_tasks == 0:
            return 0.0
        return self.failed_tasks_count / total_tasks
    
    async def _check_health_issues(self):
        """Check for health issues and take action"""
        issues = []
        
        if self.system_health.cpu_usage > 90:
            issues.append("High CPU usage")
        
        if self.system_health.memory_usage > 90:
            issues.append("High memory usage")
        
        if self.system_health.disk_usage > 90:
            issues.append("High disk usage")
        
        if self.system_health.network_latency > 2000:
            issues.append("High network latency")
        
        if self.system_health.error_rate > 0.1:
            issues.append("High error rate")
        
        if issues:
            logger.warning(f"Health issues detected: {', '.join(issues)}")
            await self._handle_health_issues(issues)
    
    async def _handle_health_issues(self, issues: List[str]):
        """Handle health issues"""
        for issue in issues:
            if "High CPU usage" in issue:
                # Reduce concurrent tasks
                self.max_concurrent_tasks = max(1, self.max_concurrent_tasks - 1)
                logger.info(f"Reduced max concurrent tasks to {self.max_concurrent_tasks}")
            
            elif "High memory usage" in issue:
                # Clear completed tasks from memory
                self._cleanup_completed_tasks()
                logger.info("Cleaned up completed tasks from memory")
            
            elif "High error rate" in issue:
                # Pause task processing temporarily
                await self._pause_task_processing()
                logger.info("Paused task processing due to high error rate")
    
    async def _process_pending_tasks(self):
        """Process pending tasks"""
        pending_tasks = [
            task_id for task_id, task in self.tasks.items()
            if task.status == "pending" and task_id not in self.active_tasks
        ]
        
        # Sort by priority (higher first)
        pending_tasks.sort(key=lambda x: self.tasks[x].priority, reverse=True)
        
        # Start tasks up to max concurrent limit
        for task_id in pending_tasks:
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                break
            
            task = self.tasks[task_id]
            task.status = "in_progress"
            task.assigned_to = "focus_agent"
            self.active_tasks.add(task_id)
            
            # Start task processing
            asyncio.create_task(self._process_task(task_id))
    
    async def _process_task(self, task_id: str):
        """Process a specific task"""
        task = self.tasks[task_id]
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing task: {task_id} - {task.description}")
            
            # Process based on task type
            if task.task_type == "token_scan":
                result = await self._process_token_scan_task(task)
            elif task.task_type == "ai_analysis":
                result = await self._process_ai_analysis_task(task)
            elif task.task_type == "trade_execution":
                result = await self._process_trade_execution_task(task)
            elif task.task_type == "risk_check":
                result = await self._process_risk_check_task(task)
            elif task.task_type == "performance_review":
                result = await self._process_performance_review_task(task)
            else:
                result = {"error": f"Unknown task type: {task.task_type}"}
            
            # Update task status
            await self.update_task_status(task_id, "completed", result)
            
            # Update performance metrics
            completion_time = (datetime.now() - start_time).total_seconds()
            self._update_task_performance_metrics(completion_time)
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _process_token_scan_task(self, task: Task) -> Dict[str, Any]:
        """Process token scan task"""
        # This would typically call the market data manager
        # For now, return a mock result
        return {
            "tokens_found": 5,
            "scan_duration": 2.5,
            "new_opportunities": 2
        }
    
    async def _process_ai_analysis_task(self, task: Task) -> Dict[str, Any]:
        """Process AI analysis task"""
        # This would typically call the chat agent
        # For now, return a mock result
        return {
            "analysis_completed": True,
            "confidence_score": 0.85,
            "recommendation": "buy"
        }
    
    async def _process_trade_execution_task(self, task: Task) -> Dict[str, Any]:
        """Process trade execution task"""
        # This would typically call the sniper bot
        # For now, return a mock result
        return {
            "trade_executed": True,
            "execution_time": 1.2,
            "slippage": 0.02
        }
    
    async def _process_risk_check_task(self, task: Task) -> Dict[str, Any]:
        """Process risk check task"""
        # This would typically call the risk manager
        # For now, return a mock result
        return {
            "risk_assessment": "medium",
            "risk_factors": ["low_liquidity", "high_volatility"],
            "recommendation": "proceed_with_caution"
        }
    
    async def _process_performance_review_task(self, task: Task) -> Dict[str, Any]:
        """Process performance review task"""
        # This would typically analyze system performance
        # For now, return a mock result
        return {
            "performance_score": 0.92,
            "optimization_suggestions": ["increase_cache_size", "optimize_api_calls"],
            "system_health": "good"
        }
    
    async def _monitor_active_tasks(self):
        """Monitor active tasks for timeouts"""
        current_time = datetime.now()
        
        for task_id in list(self.active_tasks):
            task = self.tasks[task_id]
            
            # Check for timeout
            if task.deadline and current_time > task.deadline:
                logger.warning(f"Task {task_id} timed out")
                await self.update_task_status(task_id, "failed", {"error": "timeout"})
            
            # Check for stuck tasks (no update for too long)
            elif (current_time - task.created_at).total_seconds() > self.task_timeout:
                logger.warning(f"Task {task_id} appears stuck")
                await self.update_task_status(task_id, "failed", {"error": "stuck"})
    
    async def _update_workflow_state(self):
        """Update workflow state"""
        self.workflow_state = WorkflowState(
            current_phase=self.current_phase,
            active_tasks=list(self.active_tasks),
            completed_tasks=self.completed_tasks.copy(),
            failed_tasks=self.failed_tasks.copy(),
            performance_metrics=self.get_performance_metrics(),
            last_update=datetime.now()
        )
    
    async def _health_monitor_loop(self):
        """Health monitoring loop"""
        while self.is_running:
            try:
                await self._check_system_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(10)
    
    async def _performance_review_loop(self):
        """Performance review loop"""
        while self.is_running:
            try:
                await self._conduct_performance_review()
                await asyncio.sleep(self.performance_review_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance review loop: {e}")
                await asyncio.sleep(60)
    
    async def _conduct_performance_review(self):
        """Conduct performance review"""
        logger.info("Conducting performance review...")
        
        # Analyze task completion rates
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        if total_tasks > 0:
            success_rate = len(self.completed_tasks) / total_tasks
            logger.info(f"Task success rate: {success_rate:.2%}")
        
        # Analyze system performance
        logger.info(f"System health: CPU {self.system_health.cpu_usage:.1f}%, "
                   f"Memory {self.system_health.memory_usage:.1f}%, "
                   f"Error rate {self.system_health.error_rate:.2%}")
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions()
        if suggestions:
            logger.info(f"Optimization suggestions: {suggestions}")
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        if self.system_health.cpu_usage > 80:
            suggestions.append("Consider reducing concurrent tasks")
        
        if self.system_health.memory_usage > 80:
            suggestions.append("Consider increasing memory or optimizing data structures")
        
        if self.system_health.error_rate > 0.05:
            suggestions.append("Investigate error sources and improve error handling")
        
        if self.avg_task_completion_time > 30:
            suggestions.append("Consider optimizing task processing algorithms")
        
        return suggestions
    
    def _cleanup_completed_tasks(self):
        """Clean up completed tasks from memory"""
        # Keep only recent completed tasks
        max_completed_tasks = 100
        if len(self.completed_tasks) > max_completed_tasks:
            tasks_to_remove = self.completed_tasks[:-max_completed_tasks]
            for task_id in tasks_to_remove:
                if task_id in self.tasks:
                    del self.tasks[task_id]
            self.completed_tasks = self.completed_tasks[-max_completed_tasks:]
    
    async def _pause_task_processing(self):
        """Pause task processing temporarily"""
        # This would typically pause new task creation
        # For now, just log the action
        logger.info("Task processing paused due to health issues")
        await asyncio.sleep(30)  # Pause for 30 seconds
    
    def _update_task_performance_metrics(self, completion_time: float):
        """Update task performance metrics"""
        self.total_tasks_processed += 1
        
        # Update average completion time
        if self.avg_task_completion_time == 0:
            self.avg_task_completion_time = completion_time
        else:
            self.avg_task_completion_time = (self.avg_task_completion_time + completion_time) / 2
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        total_tasks = self.successful_tasks + self.failed_tasks_count
        success_rate = self.successful_tasks / max(total_tasks, 1)
        
        return {
            "total_tasks_processed": self.total_tasks_processed,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks_count,
            "success_rate": success_rate,
            "avg_task_completion_time": self.avg_task_completion_time,
            "active_tasks": len(self.active_tasks),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
            "uptime": (datetime.now() - self.start_time).total_seconds()
        }
    
    def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state"""
        return {
            "current_phase": self.workflow_state.current_phase,
            "active_tasks": self.workflow_state.active_tasks,
            "completed_tasks_count": len(self.workflow_state.completed_tasks),
            "failed_tasks_count": len(self.workflow_state.failed_tasks),
            "performance_metrics": self.workflow_state.performance_metrics,
            "last_update": self.workflow_state.last_update.isoformat()
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        return {
            "cpu_usage": self.system_health.cpu_usage,
            "memory_usage": self.system_health.memory_usage,
            "disk_usage": self.system_health.disk_usage,
            "network_latency": self.system_health.network_latency,
            "api_response_time": self.system_health.api_response_time,
            "error_rate": self.system_health.error_rate,
            "uptime": self.system_health.uptime,
            "timestamp": self.system_health.timestamp.isoformat()
        } 
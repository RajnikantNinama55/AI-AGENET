import logging
from app import app
from supervisor.supervisor_agent import SupervisorAgent
from flask import g

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('app')

# Initialize the supervisor agent
supervisor = None

def initialize_supervisor():
    """Initialize and start the supervisor agent"""
    global supervisor
    
    try:
        logger.info("Initializing Supervisor Agent")
        # Initialize supervisor with active strategies
        supervisor = SupervisorAgent(optimization_interval=3600)  # Run optimization every hour
        
        # Start the supervisor in a background thread
        if supervisor.start():
            logger.info("Supervisor Agent started successfully")
        else:
            logger.warning("Failed to start Supervisor Agent")
    except Exception as e:
        logger.error(f"Error initializing Supervisor Agent: {e}")

def get_supervisor_instance():
    """Get the supervisor instance, ensuring it runs in application context"""
    from supervisor.supervisor_agent import SupervisorAgent
    
    # Check if supervisor exists in the app context
    if not hasattr(g, 'supervisor'):
        # If supervisor is None, initialize it
        if supervisor is None:
            initialize_supervisor()
        
        # Store in app context
        g.supervisor = supervisor
    
    return g.supervisor

# Initialize app components
with app.app_context():
    logger.info("Application initialized successfully")
    initialize_supervisor()

# Teardown function to handle supervisor cleanup
@app.teardown_appcontext
def teardown_supervisor(exception):
    """Clean up supervisor when app context ends"""
    sup = g.pop('supervisor', None)
    if sup and sup.running:
        try:
            sup.stop()
            logger.info("Supervisor stopped during app context teardown")
        except Exception as e:
            logger.error(f"Error stopping supervisor: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

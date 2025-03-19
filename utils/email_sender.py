import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def send_violation_email(recipient_email, violations, filename):
    """
    Send email notification about detected traffic violations.
    
    Args:
        recipient_email: Email address to send notification to
        violations: List of detected violations
        filename: Original filename that was analyzed
    """
    try:
        # Get email credentials from environment variables
        smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.environ.get('SMTP_PORT', 587))
        smtp_username = os.environ.get('SMTP_USERNAME', '')
        smtp_password = os.environ.get('SMTP_PASSWORD', '')
        sender_email = os.environ.get('SENDER_EMAIL', 'traffic.violation.system@example.com')
        
        # Check if credentials are available
        if not all([smtp_username, smtp_password]):
            logger.warning("Email credentials not found. Simulating email sending.")
            # Log the email content that would have been sent
            logger.info(f"Would send email to: {recipient_email}")
            logger.info(f"Subject: Traffic Violation Detected")
            logger.info(f"Violations: {', '.join(violations)}")
            return
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = 'Traffic Violation Detected'
        
        # Email body
        body = f"Dear Sir,\n\nA traffic violation has been detected in the uploaded file: {filename}.\n\n"
        body += "Violation Type(s):\n"
        for violation in violations:
            body += f"- {violation}\n"
        body += "\nPlease review the violation and take the necessary actions.\n\n"
        body += "Regards,\nTraffic Violation Detection System"
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Violation email sent to {recipient_email}")
        
    except Exception as e:
        logger.error(f"Error sending violation email: {str(e)}", exc_info=True)
        # In a production system, you might want to queue the email for retry

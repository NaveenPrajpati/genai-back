"""SMTP email delivery.

Uses the Python standard library (`smtplib`) so no extra dependency is needed.
The blocking send runs in a worker thread to avoid stalling the async event
loop. Configure via environment variables:

    SMTP_HOST       e.g. smtp.gmail.com
    SMTP_PORT       e.g. 587 (STARTTLS) or 465 (SSL); defaults to 587
    SMTP_USER       SMTP username (login)
    SMTP_PASSWORD   SMTP password / app password
    SMTP_FROM       From address; defaults to SMTP_USER
    SMTP_USE_SSL    "true" to use implicit SSL (port 465) instead of STARTTLS
"""

import asyncio
import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def _send_sync(to: str, subject: str, body: str) -> bool:
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    if not host or not user or not password:
        logger.error("email skipped: SMTP not configured (SMTP_HOST/USER/PASSWORD)")
        return False

    port = int(os.getenv("SMTP_PORT", "587"))
    sender = os.getenv("SMTP_FROM", user)
    use_ssl = os.getenv("SMTP_USE_SSL", "").lower() in ("1", "true", "yes")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    try:
        if use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=context, timeout=15) as server:
                server.login(user, password)
                server.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(user, password)
                server.send_message(message)
        logger.info("email sent to=%s subject=%s", to, subject)
        return True
    except Exception as e:
        logger.error("email send failed to=%s: %s", to, e)
        return False


async def send_email(to: str, subject: str, body: str) -> bool:
    """Send a plain-text email. Returns True on success, never raises."""
    return await asyncio.to_thread(_send_sync, to, subject, body)

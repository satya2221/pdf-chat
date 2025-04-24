from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

def send_notification(notification_type, message):
    channel = get_channel_layer()
    async_to_sync(channel.group_send)(
        "notification",
        {
            "type": "send_notification",
            "message": {
                "type": notification_type,
                "content": message
            }
        }
    )
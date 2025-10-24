#!/bin/bash
set -e

# Default values
DEVICE_TYPE=${DEVICE_TYPE:-infusion_pump}
DEVICE_ID=${DEVICE_ID:-device_001}
MQTT_BROKER=${MQTT_BROKER:-mqtt-broker}
MQTT_PORT=${MQTT_PORT:-1883}
WARD=${WARD:-ICU}

echo "========================================="
echo "IoMT Device Simulator"
echo "========================================="
echo "Device Type:  $DEVICE_TYPE"
echo "Device ID:    $DEVICE_ID"
echo "MQTT Broker:  $MQTT_BROKER:$MQTT_PORT"
echo "Ward:         $WARD"
echo "========================================="
echo ""

# Wait for MQTT broker to be ready
echo "Waiting for MQTT broker..."
for i in {1..30}; do
    if nc -z $MQTT_BROKER $MQTT_PORT 2>/dev/null; then
        echo "✓ MQTT broker is ready"
        break
    fi
    echo "  Attempt $i/30..."
    sleep 2
done

# Launch the appropriate device simulator
case $DEVICE_TYPE in
    infusion_pump)
        echo "Starting Infusion Pump Simulator..."
        python -m devices.infusion_pump_sim.simulator \
            --device-id "$DEVICE_ID" \
            --broker "$MQTT_BROKER" \
            --port "$MQTT_PORT" \
            --ward "$WARD"
        ;;
    ecg_monitor)
        echo "Starting ECG Monitor Simulator..."
        python -m devices.ecg_monitor_sim.simulator \
            --device-id "$DEVICE_ID" \
            --broker "$MQTT_BROKER" \
            --port "$MQTT_PORT" \
            --ward "$WARD"
        ;;
    ventilator)
        echo "Starting Ventilator Simulator..."
        python -m devices.vent_sim.simulator \
            --device-id "$DEVICE_ID" \
            --broker "$MQTT_BROKER" \
            --port "$MQTT_PORT" \
            --ward "$WARD"
        ;;
    dicom)
        echo "Starting DICOM Simulator..."
        python -m devices.dicom_sim.simulator \
            --device-id "$DEVICE_ID" \
            --broker "$MQTT_BROKER" \
            --port "$MQTT_PORT" \
            --ward "$WARD"
        ;;
    fhir_gateway)
        echo "Starting FHIR Gateway Simulator..."
        python -m devices.fhir_gateway_sim.simulator \
            --device-id "$DEVICE_ID" \
            --broker "$MQTT_BROKER" \
            --port "$MQTT_PORT" \
            --ward "$WARD"
        ;;
    *)
        echo "ERROR: Unknown device type: $DEVICE_TYPE"
        exit 1
        ;;
esac

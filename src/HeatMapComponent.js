// src/HeatMapComponent.js
import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default Leaflet marker icon
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
let DefaultIcon = L.icon({ iconUrl: icon, shadowUrl: iconShadow, iconSize: [25, 41], iconAnchor: [12, 41] });
L.Marker.prototype.options.icon = DefaultIcon;

function LocationMarker({ onLocationSelect }) {
    const [position, setPosition] = useState(null);
    useMapEvents({
        click(e) {
            setPosition(e.latlng);
            onLocationSelect(e.latlng.lat, e.latlng.lng);
        },
    });
    return position === null ? null : <Marker position={position}><Popup>Selected Heat Zone</Popup></Marker>;
}

// NEW: Added userPos as a prop
const HeatMapComponent = ({ onLocationSelect, userPos }) => {
    // Center the map on the user if we have their location
    const center = userPos || [25.5358, 84.8512];

    return (
        <div style={{ height: "100%", width: "100%", borderRadius: "16px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.2)" }}>
            <MapContainer center={center} zoom={10} scrollWheelZoom={true} style={{ height: "100%", width: "100%" }}>
                {/* White/Grey Light Map */}
                <TileLayer attribution='&copy; CARTO' url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" />
                <LocationMarker onLocationSelect={onLocationSelect} />

                {/* NEW: Render the User's Location Pin */}
                {userPos && (
                    <Marker position={userPos}>
                        <Popup><strong>You are here</strong><br />Live Location</Popup>
                    </Marker>
                )}
            </MapContainer>
        </div>
    );
};

export default HeatMapComponent;
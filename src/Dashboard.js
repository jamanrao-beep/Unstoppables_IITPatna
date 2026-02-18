// src/Dashboard.js
import React from 'react';
import Navbar from './Navbar';
import { motion } from 'framer-motion';
import backgroundImage from './bckgrnd.jpg';
import './DashboardMetrics.css';

const Dashboard = () => {
    // Dummy Data
    const currentData = {
        aqi: 45,
        status: "Good",
        temp: 28,
        humidity: 65,
        pm25: 12.4,
        pm10: 28.7
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1 }}
            style={{
                width: "100%",
                position: "relative",
                overflowX: "hidden"
            }}
        >
            <Navbar />

            {/* --- BACKGROUND IMAGE LAYER --- */}
            <div style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",

                /* CHANGED: Increased height to 200vh to fit new content */
                height: "200vh",

                backgroundImage: `url(${backgroundImage})`,
                backgroundSize: "cover",
                backgroundPosition: "center",
                zIndex: -2
            }}>
                {/* Dark overlay */}
                <div style={{ position: "absolute", inset: 0, background: "rgba(0,0,0,0.5)" }} />
            </div>

            {/* --- MAIN CONTENT LAYER --- */}
            <div style={{
                position: "relative",
                zIndex: 1,
                width: "100%",
                display: "flex",
                flexDirection: "column",
                alignItems: "center"
            }}>

                {/* --- 1. METRICS SECTION (Top) --- */}
                <section className="metrics-section" style={{ height: "auto", minHeight: "100vh", paddingBottom: "50px" }}>

                    <h2 className="metrics-title">Current Environmental Metrix</h2>

                    <div className="main-row">
                        <div className="glass-box side-box left">
                            <span className="box-icon">🌡️</span>
                            <span className="box-value">{currentData.temp}°C</span>
                            <span className="box-label">Temperature</span>
                        </div>

                        <div className="aqi-circle">
                            <span className="aqi-value">{currentData.aqi}</span>
                            <span className="aqi-label">AQI</span>
                            <span className="aqi-status">{currentData.status}</span>
                        </div>

                        <div className="glass-box side-box right">
                            <span className="box-icon">💧</span>
                            <span className="box-value">{currentData.humidity}%</span>
                            <span className="box-label">Humidity</span>
                        </div>
                    </div>

                    <div className="bottom-row">
                        <div className="glass-box small-box">
                            <span className="box-label">PM 2.5</span>
                            <span className="box-value">{currentData.pm25}</span>
                        </div>
                        <div className="glass-box small-box">
                            <span className="box-label">PM 10</span>
                            <span className="box-value">{currentData.pm10}</span>
                        </div>
                    </div>


                    {/* --- 2. NEW TOURIST & HEALTH SECTION (Below Metrics) --- */}
                    <div className="suggestion-container">

                        {/* Left: Tourist Suggestion */}
                        <div className="tourist-box">
                            <div className="tourist-title">
                                <span>🧭</span> Tourist Suggestion Engine
                            </div>
                            <p className="tourist-text">
                                "The air is crisp and visibility is peak today. A sunset boat ride to the Umananda Temple or a stroll through the tea gardens is highly recommended."
                            </p>
                            <div className="icon-row">
                                <span className="icon-tag">⛴️ Ferry Travel: Optimal</span>
                                <span className="icon-tag">📸 Clear Skies</span>
                            </div>
                        </div>

                        {/* Right: Health Advisory */}
                        <div className="health-box">
                            <div className="health-title">
                                <span>❤️</span> Health Advisory
                            </div>
                            <p className="health-text">
                                Ideal conditions for elderly & children. No mask required for outdoor activities.
                            </p>
                            <span className="health-badge">Safe for All</span>
                        </div>

                    </div>

                </section>

                {/* Footer Spacer */}
                <div style={{ height: "20vh" }}></div>

            </div>

        </motion.div>
    );
};

export default Dashboard;
// src/Navbar.js
import React, { useState } from 'react';
import './Navbar.css';
import { motion } from 'framer-motion';

const Navbar = () => {
    // This state tracks which tab is currently active
    const [activeTab, setActiveTab] = useState('Overview');

    const navLinks = [
        'Overview',
        'AQI Map',
        'Heat Map',
        'About Us',
        'Eco Impact'
    ];

    return (
        <nav className="navbar">
            {/* 1. Logo & Title Section */}
            <div className="brand-container">
                {/* Placeholder for Logo - You can replace this circle with an <img> tag later */}
                <div className="logo-placeholder"></div>

                <div className="title-stack">
                    <span className="main-title">PRO ATMOS GUARD</span>
                    <span className="sub-title">IIT PATNA</span>
                </div>
            </div>

            {/* 2. Navigation Links */}
            <ul className="nav-links">
                {navLinks.map((link) => (
                    <li
                        key={link}
                        className={`nav-item ${activeTab === link ? 'active' : ''}`}
                        onClick={() => setActiveTab(link)}
                    >
                        {link}
                        {/* Animated Underline for Active Tab */}
                        {activeTab === link && (
                            <motion.div
                                className="underline"
                                layoutId="underline"
                            />
                        )}
                    </li>
                ))}
            </ul>

            {/* 3. Search Bar */}
            <div className="search-container">
                <input
                    type="text"
                    placeholder="Search location..."
                    className="search-input"
                />
                <span className="search-icon">🔍</span>
            </div>
        </nav>
    );
};

export default Navbar;
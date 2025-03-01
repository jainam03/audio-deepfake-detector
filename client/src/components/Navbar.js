// src/components/Navbar.js
import React from "react";
import { AppBar, Toolbar, Typography, IconButton } from "@mui/material";
import { Brightness4, Brightness7 } from "@mui/icons-material";
import { Link } from "react-router-dom";

const Navbar = ({ darkMode, setDarkMode }) => {
    const handleToggle = () => setDarkMode((prev) => !prev);

    return (
        <AppBar position="static">
            <Toolbar>
                <Typography
                    variant="h6"
                    component={Link}
                    to="/"
                    sx={{
                        flexGrow: 1,
                        textDecoration: "none",
                        color: "inherit",
                    }}
                >
                    Audio Deepfake Detector
                </Typography>
                <IconButton color="inherit" onClick={handleToggle}>
                    {darkMode ? <Brightness7 /> : <Brightness4 />}
                </IconButton>
            </Toolbar>
        </AppBar>
    );
};

export default Navbar;

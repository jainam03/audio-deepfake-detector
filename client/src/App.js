// // src/App.js
// import React from "react";
// import UploadForm from "./components/UploadForm";

// function App() {
//   return (
//     <div className="App">
//       <h1>Audio Deepfake Detection</h1>
//       <UploadForm />
//     </div>
//   );
// }

// export default App;

// src/App.js
import React, { useMemo, useState } from "react";
import { createTheme, ThemeProvider, CssBaseline } from "@mui/material";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import UploadForm from "./components/UploadForm";
import ResultsPage from "./components/ResultsPage";

function App() {
  const [darkMode, setDarkMode] = useState(false);

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: darkMode ? "dark" : "light",
        },
      }),
    [darkMode]
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />
        <Routes>
          <Route path="/" element={<UploadForm />} />
          <Route path="/results" element={<ResultsPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;

import React, { useState, useEffect } from 'react';
import { Navbar, Nav, Container } from 'react-bootstrap';
import WelcomePage from './pages/WelcomePage.jsx';
import InputPage from './pages/InputPage.jsx';
import OutputPage from './pages/OutputPage.jsx';
import UploadPage from './pages/UploadPage/UploadPage.jsx';
import LoginPage from './pages/LoginPage.jsx';
import SignupPage from './pages/SignupPage.jsx';
import './App.css';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import 'bootstrap/dist/css/bootstrap.min.css';

const BRAND = 'HemoSight';

function App() {
  const [viewState, setViewState] = useState('welcome');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [lastViewState, setLastViewState] = useState('welcome');

  useEffect(() => {
    const accessToken = sessionStorage.getItem('access_token');
    if (accessToken) {
      setIsLoggedIn(true);
    } else {
      setIsLoggedIn(false);
    }
  }, [viewState]);

  useEffect(() => {
    document.title = BRAND;
  }, []);

  const handleLogout = () => {
    sessionStorage.removeItem('access_token');
    advanceViewState('login');
    toast('You have been logged out. ', { type: 'success' });
  }

  const advanceViewState = (nextState) => {
    // if nextState is undefined, go back to lastViewState
    if (nextState === undefined) {
      nextState = lastViewState;
      if (nextState === 'login' || nextState === 'signup') {
        nextState = 'welcome';
      }
    }
    setLastViewState(viewState);
    setViewState(nextState);
  }

  return (
    <>
      <Navbar bg="light" expand="sm" data-bs-theme="light">
        <Navbar.Brand href="/">{BRAND}</Navbar.Brand>
        <Navbar.Toggle aria-controls="navbarNav" />
        <Navbar.Collapse id="navbarNav">
          {isLoggedIn ? (
            <>
              <Nav className="me-auto" variant="underline">
                <Nav.Link onClick={() => setViewState('input')}>Input</Nav.Link>
                <Nav.Link onClick={() => setViewState('output')}>Output</Nav.Link>
                <Nav.Link onClick={() => setViewState('upload')}>Upload</Nav.Link>
              </Nav>
              <Nav className="ms-auto">
                <Nav.Link onClick={handleLogout}>Logout</Nav.Link>
              </Nav>
            </>
          ) : (
            <>
              <Nav className="me-auto">
                <Nav.Link onClick={() => setViewState('signup')}>Sign Up</Nav.Link>
              </Nav>
              <Nav className="ms-auto">
                <Nav.Link onClick={() => setViewState('login')}>Login</Nav.Link>
              </Nav>
            </>
          )}
        </Navbar.Collapse>
      </Navbar>

      <Container className='mt-4'>
        {viewState === 'welcome' && (
          <WelcomePage />
        )}
        {viewState === 'input' && (
          <InputPage />
        )}
        {viewState === 'output' && (
          <OutputPage />
        )}
        {viewState === 'upload' && (
          <UploadPage advanceViewState={advanceViewState} />
        )}
        {viewState === 'login' && (
          <LoginPage advanceViewState={advanceViewState} />
        )}
        {viewState === 'signup' && (
          <SignupPage advanceViewState={advanceViewState} />
        )}
      </Container>

      <ToastContainer position="bottom-right"
        className="m-3"
        autoClose={5000}
        closeOnClick
        pauseOnHover
        hideProgressBar
        draggable
      >
      </ToastContainer>

      <footer>
        <Container fluid>
          <hr />
          <p className="text-center">Yuan Lab 2024 @ MD Anderson | <a href="https://www.mdanderson.org/research/departments-labs-institutes/labs/yuan-laboratory.html" 
          target="_blank" rel="noreferrer">Yuan Lab</a></p>
        </Container>
      </footer>
    </>
  );
}

export { BRAND, App };

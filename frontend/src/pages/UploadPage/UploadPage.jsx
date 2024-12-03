import React, { useState, useEffect, useRef } from 'react';
import { Button, Container, Card, Row, Col, Stack } from 'react-bootstrap';
import { toast } from 'react-toastify';
import { FigureCard, fetchVegaPlot } from '../../components/FigureCard.jsx';
import OpenProjectModal from './OpenProjectModal.jsx';
import NewProjectModal from './NewProjectModal.jsx';
import DeleteProjectModal from './DeleteProjectModal.jsx';
import Explorer from './Explorer.jsx';

function UploadPage({ advanceViewState }) {
  const [project, setProject] = useState(null);
  const [user, setUser] = useState('');
  const [ws, setWs] = useState(null);

  const fetchUser = async () => {
    const accessToken = sessionStorage.getItem('access_token');
    if (!accessToken) {
      return;
    }
    try {
      const response = await fetch('/api/current_user', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json'
        }
      });
      if (!response.ok) {
        // 401 Unauthorized
        // clear access token
        sessionStorage.removeItem('access_token');
        advanceViewState('login');
        toast('Session expired. Please login again.', { type: 'error' })
      }
      const data = await response.json();
      setUser(data.username);
    }
    catch (error) {
      console.error('Error fetching user:', error);
    }
  };

  const openSocket = () => {
    const wsProtocol = window.location.protocol === "https:" ? "wss://" : "ws://";
    const wsUrl = `${wsProtocol}${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);
    socket.onopen = () => {
      console.log('WebSocket connected');
      socket.send(JSON.stringify({ command: 'subscribe', argument: { project_id: project.id } }));
    };
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('WebSocket message received:', data);
      if (data['command'] === 'predict') {
        console.log('Predict:', data['argument']);
        const image_id = data['argument']['image_id'];
        ExplorerRef.current.updateImage(image_id);
      }
    };
    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    socket.onclose = () => {
      console.log('WebSocket connection closed');
    };
    setWs(socket);
  };

  useEffect(() => {
    fetchUser();
    fetchProjects();
    if (project !== null) {
      openSocket();
    }
    return () => {
      if (ws !== null) {
        ws.close();
      }
    };
  }, [project]);

  const exitProject = () => {
    setProject(null);
    setFigures([]);
  };

  const [projects, setProjects] = useState([]);
  const fetchProjects = async () => {
    const accessToken = sessionStorage.getItem('access_token');
    if (accessToken) {
      const response = await fetch(`/api/project`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });
      const data = await response.json();
      setProjects(data);
    }
  };

  const [figures, setFigures] = useState([]);

  const plotTSNEQuery = async () => {
    let plot = await fetchVegaPlot('tsne_query.lite');
    plot.data.url = `/api/tsne_query?project_id=${project.id}`;
    return plot;
  };

  const loadFigure = async () => {
    setFigures(() => []);
    const newFigure = {
      plot: await plotTSNEQuery(),
      title: 't-SNE Query',
      index: 1
    };
    setFigures(prevFigures => [...prevFigures, newFigure]);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const DeleteProjectModalRef = useRef(null);
  const OpenProjectModalRef = useRef(null);
  const NewProjectModalRef = useRef(null);
  const ExplorerRef = useRef(null);

  return (
    <>
      {project === null ? (
        <Container>
          <Card className='mt-4'>
            <Card.Header as="h5">AI Prediction</Card.Header>
            <Card.Body>
              <p>Welcome {user}!</p>
              <p>Please create or retrieve a project: </p>
              <Button onClick={() => NewProjectModalRef.current.show()}>
                Create Project
              </Button>{' '}
              <Button onClick={() => OpenProjectModalRef.current.show()}
                disabled={projects.length === 0}>
                Open Project
              </Button>
            </Card.Body>
          </Card>
        </Container>
      ) : (
        <Container>
          <Card className='mt-4'>
            <Card.Header as="h5">AI Prediction: Project -- {project.name}</Card.Header>
            <Card.Body>
              <Card.Text>
                Welcome {user}! Project created time: {formatDate(project.created_at)}.
              </Card.Text>
              <Button onClick={exitProject}>
                Exit Project
              </Button>{' '}
              <Button onClick={() => DeleteProjectModalRef.current.show()}
                variant='danger'>
                Delete Project
              </Button>
            </Card.Body>
          </Card>
          <Card className='mt-4'>
            <Card.Header as="h5">Data Explorer</Card.Header>
            <Card.Body>
              <Stack gap={2}>
                <Button onClick={loadFigure}>
                  Load Global t-SNE Scatter Plot
                </Button>
                <Explorer
                  project={project}
                  ref={ExplorerRef} />
              </Stack>
            </Card.Body>
          </Card>
          {figures.map((figure, index) => (
            <FigureCard key={index}
              title={figure.title}
              plot={figure.plot}
              index={figure.index} />
          ))}
        </Container>
      )}

      <NewProjectModal
        ref={NewProjectModalRef}
        setProject={setProject}
      />
      <OpenProjectModal
        ref={OpenProjectModalRef}
        setProject={setProject}
      />
      <DeleteProjectModal
        ref={DeleteProjectModalRef}
        project={project}
        setProject={setProject}
      />
    </>
  )
}

export default UploadPage;

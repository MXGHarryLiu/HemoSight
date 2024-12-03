import React, { useState } from 'react';
import { Modal, Button, Form, Container, Card } from 'react-bootstrap';
import { FigureCard, fetchVegaPlot } from '../components/FigureCard.jsx';

function OutputPage() {
  const [runs, setRuns] = useState([]);
  const [models, setModels] = useState([]);
  const [run, setRun] = useState(null);
  const [model, setModel] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [figures, setFigures] = useState([]);

  const submitForm = async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    setShowModal(false);
    const selectedRun = form.elements.run.value;
    const selectedModel = form.elements.model.value;
    setRun(selectedRun);
    setModel(selectedModel);
    setFigures(() => []);
    let newFigure = {
      plot: await plotLoss(selectedRun),
      title: 'Training progress',
      index: 1
    };
    setFigures(prevFigures => [...prevFigures, newFigure]);
    newFigure = {
      plot: await plotCM(selectedRun, selectedModel),
      title: 'Confusion matrix',
      index: 2
    };
    setFigures(prevFigures => [...prevFigures, newFigure]);
    newFigure = {
      plot: await plotAccuracy(selectedRun, selectedModel),
      title: 'Accuracy vs. Proportion of Labeled Data',
      index: 3
    };
    setFigures(prevFigures => [...prevFigures, newFigure]);
    newFigure = {
      plot: await plotTSNE(selectedRun, selectedModel),
      title: 't-SNE',
      index: 4
    };
    setFigures(prevFigures => [...prevFigures, newFigure]);
  };

  const plotLoss = async (selectedRun) => {
    let plot = await fetchVegaPlot('loss.lite');
    plot.data.url = `/api/loss?run=${selectedRun}`;
    return plot;
  };
  const plotCM = async (selectedRun, selectedModel) => {
    let plot = await fetchVegaPlot('cm');
    plot.data[0].url = `/api/confusion_matrix?run=${selectedRun}&model=${selectedModel}`;
    return plot;
  };
  const plotAccuracy = async (selectedRun, selectedModel) => {
    let plot = await fetchVegaPlot('accuracy.lite');
    plot.data.url = `/api/accuracy?run=${selectedRun}&model=${selectedModel}`;
    return plot;
  };
  const plotTSNE = async (selectedRun, selectedModel) => {
    let plot = await fetchVegaPlot('tsne.lite');
    plot.data.url = `/api/tsne?run=${selectedRun}&model=${selectedModel}`;
    return plot;
  };

  const fetchRuns = async () => {
    try {
      const response = await fetch(`/api/run`);
      const data = await response.json();
      setRuns(data);
      if (data.length > 0) {
        fetchModels(data[0]);
      }
    } catch (error) {
      console.error('Error fetching run options:', error);
    }
  };

  const fetchModels = async (currentRun) => {
    try {
      const response = await fetch(`/api/model?run=${currentRun}`);
      const data = await response.json();
      setModels(data);
    } catch (error) {
      console.error('Error fetching model options:', error);
    }
  };

  const handleShow = () => {
    fetchRuns();
    setShowModal(true);
  };
  const handleRunClose = () => setShowModal(false);
  const handleRunChange = (event) => fetchModels(event.target.value);

  return (
    <>
      <Container>
        <h2 className="text-center">Output</h2>
        <Card className='mt-4'>
          <Card.Header as="h5">Input</Card.Header>
          <Card.Body>
            <Card.Text>
              {run ? `Selected run: ${run}` : 'Please select a run. '}<br/>
              {model ? `Selected model: ${model}` : 'Please select a model. '}
            </Card.Text>
            <Button onClick={handleShow}>
              Select a model
            </Button>
          </Card.Body>
        </Card>
        {figures.map((figure, index) => (
          <FigureCard key={index} 
          title={figure.title} 
          plot={figure.plot} 
          index={figure.index} />
        ))}
      </Container>

      <Modal show={showModal} onHide={handleRunClose}>
        <Modal.Header closeButton>
          <Modal.Title>Please select a run</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form id="runForm"
            onSubmit={submitForm}>
            <Form.Group className="mb-2">
              <Form.Label>Select a run</Form.Label>
              <Form.Select
                name="run"
                autoFocus
                onChange={handleRunChange}>
                {runs.map((run, index) => (
                  <option key={index} value={run}>
                    {run}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
            <Form.Group className="mb-2">
              <Form.Label>Select a model</Form.Label>
              <Form.Select
                name="model"
                >
                {models.map((model, index) => (
                  <option key={index} value={model}>
                    {model}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </Form>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={handleRunClose}>
            Close
          </Button>
          <Button variant="primary" type="submit" form="runForm">
            Submit
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
}

export default OutputPage;

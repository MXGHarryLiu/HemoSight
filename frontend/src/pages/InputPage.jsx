import React, { useState } from 'react';
import { FigureCard, fetchVegaPlot} from '../components/FigureCard.jsx';
import { Modal, Button, Form, Container, Card } from 'react-bootstrap';

function InputPage() {
  const [datasets, setDatasets] = useState([]);
  const [dataset, setDataset] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [figures, setFigures] = useState([]);

  const submitForm = async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    setShowModal(false);
    const selectedDataset = form.elements.dataset.value;
    setDataset(selectedDataset);
    setFigures(() => []);
    const newFigure = {
      plot: await plotCount(selectedDataset),
      title: 'Label distribution bar chart',
      index: 1
    };
    setFigures(prevFigures => [...prevFigures, newFigure]);
  };

  const plotCount = async (selectedDataset) => {
    let plot = await fetchVegaPlot('label.lite');
    plot.data.url = `/api/label_data?dataset=${selectedDataset}`;
    return plot;
  };

  const fetchDatasets = async () => {
    try {
      const response = await fetch(`/api/dataset`);
      const data = await response.json();
      setDatasets(data);
    } catch (error) {
      console.error('Error fetching dataset options:', error);
    }
  };
  
  const handleShow = () => {
    fetchDatasets();
    setShowModal(true);
  };
  const handleClose = () => setShowModal(false);

  return (
    <>
      <Container>
        <h2 className="text-center">Input Data</h2>
        <Card className='mt-4'>
          <Card.Header as="h5">Input</Card.Header>
          <Card.Body>
            <Card.Text>
              {dataset ? `Selected dataset: ${dataset}` : 'Please select a dataset. '}
            </Card.Text>
            <Button onClick={handleShow} variant="primary">
              Select a dataset
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

      <Modal show={showModal} onHide={handleClose}>
        <Modal.Header closeButton>
          <Modal.Title>Please select a dataset</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form id="datasetForm" onSubmit={submitForm}>
            <Form.Group className="mb-2">
              <Form.Select
                name="dataset"
                autoFocus
              >
                {datasets.map((dataset, index) => (
                  <option key={index} value={dataset}>
                    {dataset}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </Form>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="primary" type="submit" form="datasetForm">
            Submit
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
}

export default InputPage;

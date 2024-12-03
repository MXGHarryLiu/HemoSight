import React from 'react';
import { Modal, Button, Form } from 'react-bootstrap';

class OpenProjectModal extends React.Component {
  // props: setProject
  constructor(props) {
    super(props);
    this.state = {
      showModal: false, 
      projects: []
    };
  }

  show = () => {
    this.setState({ showModal: true });
  };

  #hide = () => {
    this.setState({ showModal: false });
  };

  #onSubmit = (event) => {
    const { setProject } = this.props;
    const { projects } = this.state;
    event.preventDefault();
    const form = event.currentTarget;
    const selectedProjectID = form.elements.name.value;
    const selectedProject = projects[selectedProjectID];
    setProject(selectedProject);
    this.#hide();
  };

  componentDidUpdate(prevProps, prevState) {
    if (this.state.showModal && !prevState.showModal) {
      this.#fetchProjects();
    }
  }

  #fetchProjects = async () => {
    const accessToken = sessionStorage.getItem('access_token');
    if (accessToken) {
      const response = await fetch(`/api/project`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });
      const data = await response.json();
      this.setState({ projects: data });
    }
  };

  render() {
    const { showModal, projects } = this.state;

    return (
      <Modal show={showModal} onHide={this.#hide}>
        <Modal.Header closeButton>
          <Modal.Title>Open Project</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form noValidate
            id="openProjectForm"
            onSubmit={this.#onSubmit}>
            <Form.Group>
              <Form.Label>Name</Form.Label>
              <Form.Select
                name='name'
                autoFocus
              >
                {projects.map((s, index) => (
                  <option key={index} value={index}>
                    {s.name}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </Form>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="primary" type="submit" form="openProjectForm">
            Open
          </Button>
        </Modal.Footer>
      </Modal>
    );
  }
}

export default OpenProjectModal;

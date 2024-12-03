import React from 'react';
import { Modal, Button, Form } from 'react-bootstrap';
import { Formik } from 'formik';
import * as Yup from "yup";
import { toast } from 'react-toastify';

class NewProjectModal extends React.Component {
  // props: setProject
  constructor(props) {
    super(props);
    this.state = {
      showModal: false, 
    };
  }

  show = () => {
    this.setState({ showModal: true });
  };

  #hide = () => {
    this.setState({ showModal: false });
  };

  #onSubmit = async (values) => {
    const accessToken = sessionStorage.getItem('access_token');
    if (!accessToken) {
      return;
    }
    const { setProject } = this.props;
    const queryParams = new URLSearchParams({
      name: values.name
    });
    const response = await fetch(`/api/project?${queryParams}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${accessToken}`
      }
    });
    const data = await response.json();
    setProject(data);
    toast(`Project created: ${values.name}`, { type: 'success' });
    this.#hide();
  }

  render() {
    const { showModal } = this.state;

    return (
      <Modal show={showModal} onHide={this.#hide}>
        <Modal.Header closeButton>
          <Modal.Title>Create Project</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Formik initialValues={{ name: '' }}
            validationSchema={Yup.object({
              name: Yup.string()
                .required("Required")
                .max(64, "Must be 64 characters or less")
                .matches(/^[a-zA-Z0-9]+$/, "Must contain only letters and numbers")
            })}
            onSubmit={this.#onSubmit}
          >
            {(formik) => (
              <Form noValidate
                id="newProjectForm"
                onSubmit={formik.handleSubmit}
              >
                <Form.Group controlId="newProjectName">
                  <Form.Label>Name</Form.Label>
                  <Form.Control
                    name='name'
                    type="text"
                    autoFocus
                    onChange={formik.handleChange}
                    onBlur={formik.handleBlur}
                    value={formik.values.name}
                    isInvalid={!!formik.errors.name && formik.touched.name}
                  />
                  <Form.Control.Feedback type="invalid">
                    {formik.errors.name}
                  </Form.Control.Feedback>
                </Form.Group>
              </Form>
            )}
          </Formik>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="primary" type="submit" form="newProjectForm">
            Create
          </Button>
        </Modal.Footer>
      </Modal>
    );
  }
}

export default NewProjectModal;

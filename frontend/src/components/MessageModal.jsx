import React from 'react';
import { Modal, Button } from 'react-bootstrap';

class MessageModal extends React.Component {
  // props: title, message, onConfirm
  constructor(props) {
    super(props);
    this.state = {
      showModal: false
    };
  }

  show = () => {
    this.setState({ showModal: true });
  }

  #hide = () => {
    this.setState({ showModal: false });
  }

  #onSubmit = () => {
    this.props.onConfirm();
    this.#hide();
  }

  render() {
    const { title, message } = this.props;
    const { showModal } = this.state;

    return (
      <Modal show={showModal} onHide={this.#hide}>
        <Modal.Header closeButton>
          <Modal.Title>{title}</Modal.Title>
        </Modal.Header>
        <Modal.Body dangerouslySetInnerHTML={{ __html: message }} />
        <Modal.Footer>
          <Button variant="secondary" onClick={this.#hide}>
            Cancel
          </Button>
          <Button variant="primary" onClick={this.#onSubmit}>
            Confirm
          </Button>
        </Modal.Footer>
      </Modal>
    );
  }
}

export default MessageModal;

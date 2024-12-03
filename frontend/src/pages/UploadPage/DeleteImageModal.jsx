import React from 'react';
import MessageModal from '../../components/MessageModal.jsx';
import { toast } from 'react-toastify';

class DeleteImageModal extends React.Component {
  // props: parent
  constructor(props) {
    super(props);
    this.state = {
      image_id: ''
    };
    this.MessageModalRef = React.createRef();
  }

  shouldComponentUpdate(nextProps, nextState) {
    // only update if image_id changes
    return this.state.image_id !== nextState.image_id;
  }

  show = (image_id) => {
    this.setState({ image_id: image_id });
    this.MessageModalRef.current.show();
  };

  #onSubmit = async () => {
    const { parent } = this.props;
    const { image_id } = this.state;
    const response = await fetch(`/api/query_image?image_id=${image_id}`, {
      method: 'DELETE'
    });
    if (response.ok) {
      toast('One image deleted', { type: 'success' });
      parent.deleteSelectedImageFinished(image_id);
    } else {
      toast('Failed to delete image', { type: 'error' });
    }
  };

  render() {
    const { parent } = this.props;
    const { image_id } = this.state;
    let image_name = '';
    if (image_id !== '') {
      const selectedImage = parent.state.images.find(image => image.image_id === image_id);
      image_name = selectedImage.filename;
    }
    
    return (
      <MessageModal
        ref={this.MessageModalRef}
        onConfirm={this.#onSubmit}
        title="Delete Image"
        message={`Are you sure you want to delete image ${image_name}?<br>
          This action cannot be undone.`}
      />
    );
  }
}

export default DeleteImageModal;

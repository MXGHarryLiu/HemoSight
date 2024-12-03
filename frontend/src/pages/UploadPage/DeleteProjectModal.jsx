import React from 'react';
import MessageModal from '../../components/MessageModal.jsx';
import { toast } from 'react-toastify';

class DeleteProjectModal extends React.Component {
  // props: project, setProject
  constructor(props) {
    super(props);
    this.MessageModalRef = React.createRef();
  }

  show = () => {
    this.MessageModalRef.current.show();
  };

  #onSubmit = async () => {
    const accessToken = sessionStorage.getItem('access_token');
    const { project, setProject } = this.props;
    if (!accessToken) {
      return;
    }
    const response = await fetch(`/api/project/${project.id}`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${accessToken}`
      }
    });
    if (response.ok) {
      toast(`Project deleted: ${project.name}`, { type: 'success' });
      setProject(null);
    }
  };

  render() {
    const { project } = this.props;
    
    return (
      <MessageModal
        ref={this.MessageModalRef}
        onConfirm={this.#onSubmit}
        title="Delete Project"
        message={`Are you sure you want to delete project ${project ? project.name : ''}?`}
      />
    );
  }
}

export default DeleteProjectModal;

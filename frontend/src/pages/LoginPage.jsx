import React from 'react';
import { Form, Button, Container } from 'react-bootstrap';
import { toast } from 'react-toastify';
import { Formik } from 'formik';
import * as Yup from 'yup';

function LoginPage({ advanceViewState }) {
  const handleLoginSubmit = async (values) => {
    const username = values.username;
    try {
      const response = await fetch('/api/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({
          username: username,
          password: values.password
        })
      });
      const data = await response.json();
      if (response.ok) {
        sessionStorage.setItem('access_token', data.access_token);
        toast(`Welcome ${username}!`, { type: 'success' });
        advanceViewState();
      } else {
        //toast('Incorrect username or password');
        toast(data.detail, { type: 'error' });
      }
    } catch (error) {
      toast(`${error.message}`, { type: 'error' });
    }
  };

  return (
    <Container style={{ maxWidth: "400px" }}>
      <h2 className="text-center">Login</h2>
      <Formik
        initialValues={{
          username: "",
          password: ""
        }}
        validationSchema={Yup.object({
          username: Yup.string()
            .required("Please enter your username"),
          password: Yup.string()
            .required("Please enter your password")
        })}
        onSubmit={handleLoginSubmit}
      >
        {(formik) => (
          <Form id="loginForm"
            noValidate
            onSubmit={formik.handleSubmit}
            className="mt-4">
            <Form.Group controlId="username">
              <Form.Label>Username</Form.Label>
              <Form.Control
                type="text"
                name="username"
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                value={formik.values.username}
                isInvalid={!!formik.errors.username && formik.touched.username}
                autoFocus />
              <Form.Control.Feedback type="invalid">
                {formik.errors.username}
              </Form.Control.Feedback>
            </Form.Group>
            <Form.Group controlId="password">
              <Form.Label>Password</Form.Label>
              <Form.Control
                type="password"
                name="password"
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                value={formik.values.password}
                isInvalid={!!formik.errors.password && formik.touched.password}
                required />
              <Form.Control.Feedback type="invalid">
                {formik.errors.password}
              </Form.Control.Feedback>
            </Form.Group>
            <Button variant="primary" type="submit" form="loginForm">
              Login
            </Button>
          </Form>
        )}
      </Formik>
    </Container>
  );
}

export default LoginPage;

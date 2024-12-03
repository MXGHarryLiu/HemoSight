import React from 'react';
import { Form, Button, Container } from 'react-bootstrap';
import { toast } from 'react-toastify';
import { Formik } from 'formik';
import * as Yup from "yup";

function SignupPage({ advanceViewState }) {
  const handleFormSubmit = async (values) => {
    try {
      const response = await fetch('/api/user', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(values)
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail);
      }
      toast(`User created ${values.username}! Please login. `, { type: 'success' });
      advanceViewState('login');
    } catch (error) {
      toast(`${error.message}`, { type: 'error' });
    }
  };

  return (
    <Container style={{ maxWidth: "400px" }}>
      <h2 className="text-center">Sign Up</h2>
      <Formik
        initialValues={{
          username: "",
          email: "",
          password: "",
          confirmPassword: ""
        }}
        validationSchema={Yup.object({
          username: Yup.string()
            .required("Required")
            .max(64, "Must be 64 characters or less")
            .matches(/^[a-zA-Z0-9]+$/, "Must contain only letters and numbers"),
          email: Yup.string()
            .email("Invalid email address")
            .required("Required"),
          password: Yup.string()
            .required("Required")
            .min(6, "Must be 6 characters or more")
            .matches(/(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{6,}/, "Must contain at least one uppercase letter, one lowercase letter, and one digit"),
          confirmPassword: Yup.string()
            .required("Required")
            .oneOf([Yup.ref("password"), null], "Passwords must match")
        })}
        onSubmit={handleFormSubmit}
      >
        {(formik) => (
          <Form id="signupForm"
            noValidate
            onSubmit={formik.handleSubmit}
          >
            <Form.Group controlId="username">
              <Form.Label>Username</Form.Label>
              <Form.Control
                type="text"
                name="username"
                autoFocus
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                value={formik.values.username}
                isInvalid={!!formik.errors.username && formik.touched.username}
              />
              <Form.Control.Feedback type="invalid">
                {formik.errors.username}
              </Form.Control.Feedback>
            </Form.Group>
            <Form.Group controlId="email">
              <Form.Label>Email</Form.Label>
              <Form.Control
                type="email"
                name="email"
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                value={formik.values.email}
                isInvalid={!!formik.errors.email && formik.touched.email}
              />
              <Form.Control.Feedback type="invalid">
                {formik.errors.email}
              </Form.Control.Feedback>
            </Form.Group>
            <Form.Group>
              <Form.Label>Password</Form.Label>
              <Form.Control
                type="password"
                name="password"
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                value={formik.values.password}
                isInvalid={!!formik.errors.password && formik.touched.password}
              />
              <Form.Control.Feedback type="invalid">
                {formik.errors.password}
              </Form.Control.Feedback>
            </Form.Group>
            <Form.Group>
              <Form.Label>Confirm Password</Form.Label>
              <Form.Control
                type="password"
                name="confirmPassword"
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                value={formik.values.confirmPassword}
                isInvalid={!!formik.errors.confirmPassword && formik.touched.confirmPassword}
              />
              <Form.Control.Feedback type="invalid">
                {formik.errors.confirmPassword}
              </Form.Control.Feedback>
            </Form.Group>
            <Button variant="primary" type="submit" form="signupForm">
              Sign up
            </Button>
          </Form>
        )}
      </Formik>
    </Container>
  );
}

export default SignupPage;
